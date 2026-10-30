#include "gpu-activity.h"
#include "activity-intervals.h"
#include "gpu-op-tags.h"
#include <array>
#include <cupti.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

namespace gpu_activity {
namespace {
constexpr uint64_t SECOND = 1000000000ULL;
constexpr uint64_t KEEP = 60, LAG = 5;
// Keep a full minute of production kernels, including the higher record rate
// during large-model prefill. Kernel categories retain at most one extra interval per kernel, so the interval
// storage bound is at most about488MiB before vector
// capacity overhead, bounded independently from CUPTI's callback buffers.
constexpr size_t BUFFER = 8 * 1024 * 1024, MAX_BUFFERS = 16, MAX_RECORDS = 16000000;
constexpr uint64_t MAX_FILE = 768ULL * 1024 * 1024;
uint64_t wall_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}
uint64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct record { uint64_t start, end; uint32_t device, kind, activity_kind, copy_kind, operation; uint64_t transfer_bytes; };
struct device_window {
    intervals kernel, transfer, matvec, bulk_matmul, recurrent;
    uint64_t kernel_parts = 0, transfer_parts = 0;
    std::array<uint64_t, operation_count> operation_ns{}, operation_parts{};
};
struct window {
    std::map<uint32_t, device_window> devices;
    uint64_t revision = 0, written = 0;
};
struct collector {
    void * library = nullptr;
    decltype(&cuptiGetTimestamp) timestamp = nullptr;
    decltype(&cuptiActivityRegisterCallbacks) callbacks = nullptr;
    decltype(&cuptiActivityEnable) enable = nullptr;
    decltype(&cuptiActivityDisable) disable = nullptr;
    decltype(&cuptiActivityGetNextRecord) next = nullptr;
    decltype(&cuptiActivityGetNumDroppedRecords) dropped = nullptr;
    decltype(&cuptiActivityFlushAll) flush = nullptr;
    std::atomic<bool> running{false}, stopping{false}, recording{false};
    std::atomic<uint64_t> buffers{0}, records{0}, drop_count{0}, invalid{0}, errors{0}, late{0};
    std::atomic<uint64_t> callback_ns{0}, flush_ns{0}, bytes{0}, windows_written{0};
    std::atomic<int64_t> state_clock_offset{0};
    std::atomic<uint64_t> calibration_steady{0}, state_wait_ns{0}, state_wait_max_ns{0};
    std::atomic<uint64_t> mode{0}, pending{0};
    std::mutex mutex, wait_mutex;
    std::condition_variable cv;
    std::map<uint64_t, window> windows;
    std::vector<record> invalid_examples;
    uint64_t invalid_examples_taken = 0;
    std::map<uint64_t, uint64_t> states;
    std::map<uint64_t,bool> capture_states;
    std::set<uint32_t> devices;
    std::thread worker;
    uint64_t started = 0, oldest = 0, next_window = 0;
    std::atomic<uint64_t> duty_on{0},duty_period{0};
    uint64_t transition_guard=2*SECOND,until_wall_ns=0,expired_at=0;
    std::string stop_path;
    int fd = -1;

    uint64_t now() {
        uint64_t result = 0;
        if (!timestamp || timestamp(&result) != CUPTI_SUCCESS) { ++errors; }
        return result;
    }
    void write(const std::string & line) {
        if (fd < 0) { return; }
        if (bytes + line.size() + 1 > MAX_FILE) { ++errors; ::close(fd); fd = -1; return; }
        std::string value = line + '\n'; size_t done = 0;
        while (done < value.size()) {
            const ssize_t n = ::write(fd, value.data() + done, value.size() - done);
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) { ++errors; ::close(fd); fd = -1; return; }
            done += size_t(n);
        }
        bytes += done;
    }
    void fault(const char * kind, uint64_t count) {
        std::ostringstream out;
        out << "{\"type\":\"quality\",\"time_ns\":" << now() << ",\"wall_ns\":" << wall_ns()
            << ",\"kind\":\"" << kind << "\",\"count\":" << count << "}";
        write(out.str());
    }
    void append(const std::vector<record> & rows) {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto & r : rows) {
            ++records;
            if (!r.start || r.end <= r.start || r.end-r.start > 120*SECOND || r.device > 255) {
                ++invalid;
                // Keep bounded scalar evidence. Report only on the collector
                // thread so callback I/O cannot stall source-state updates.
                if (invalid_examples_taken < 16) {
                    invalid_examples.push_back(r);
                    ++invalid_examples_taken;
                }
                continue;
            }
            devices.insert(r.device);
            const uint64_t first = r.start / SECOND, last = (r.end - 1) / SECOND;
            if (first < oldest) { ++late; continue; }
            if (pending > MAX_RECORDS) { ++errors; continue; }
            for (uint64_t b = first; b <= last; ++b) {
                auto & w = windows[b]; auto & d = w.devices[r.device];
                interval part{std::max(r.start, b*SECOND), std::min(r.end, (b+1)*SECOND)};
                if (r.kind == 0) {
                    d.kernel.push_back(part); ++d.kernel_parts;
                    const auto op = r.operation < operation_count ? r.operation : 0;
                    d.operation_ns[op] += part.end - part.start; ++d.operation_parts[op];
                    if (op == 1) d.matvec.push_back(part);
                    else if (op == 2 || op == 3) d.bulk_matmul.push_back(part);
                    else if (op == 5) d.recurrent.push_back(part);
                } else { d.transfer.push_back(part); ++d.transfer_parts; }
                ++w.revision; ++pending;
            }
        }
    }
    struct segment { uint64_t start, end, mode; };
    intervals coverage(uint64_t start,uint64_t end,uint64_t now_time) {
        intervals result;
        for(auto it=capture_states.begin();it!=capture_states.end();++it) {
            if(!it->second)continue;
            auto next_it=std::next(it);
            const uint64_t finish=next_it==capture_states.end()?now_time:next_it->first;
            const uint64_t lo=std::max(start,it->first+transition_guard);
            const uint64_t hi=std::min(end,finish>transition_guard?finish-transition_guard:0);
            if(hi>lo)result.push_back({lo,hi});
        }
        return unite(std::move(result));
    }
    std::vector<segment> segments(uint64_t start, uint64_t end) {
        std::vector<segment> result;
        auto it = states.upper_bound(start);
        uint64_t current = it == states.begin() ? 0 : std::prev(it)->second;
        uint64_t pos = start;
        for (; it != states.end() && it->first < end; ++it) {
            if (it->first > pos) { result.push_back({pos, it->first, current}); }
            current = it->second; pos = std::max(pos, it->first);
        }
        if (pos < end) { result.push_back({pos, end, current}); }
        return result;
    }
    std::string render(uint64_t b, const window & w, const std::vector<segment> & pieces,
            const intervals & covered, const std::set<uint32_t> & known_devices) {
        const uint64_t start = b*SECOND, end = start+SECOND;
        std::ostringstream out;
        out << "{\"type\":\"window\",\"start_ns\":" << start << ",\"end_ns\":" << end
            << ",\"revision\":" << w.revision << ",\"partial\":" << (start<started ? "true":"false")
            << ",\"coverage_ns\":" << length(covered)
            << ",\"dropped_total\":" << drop_count << ",\"invalid_total\":" << invalid
            << ",\"errors_total\":" << errors << ",\"late_total\":" << late << ",\"devices\":[";
        bool first_device = true;
        std::map<uint32_t, intervals> all_kernels;
        for (uint32_t id : known_devices) {
            const auto found = w.devices.find(id);
            const device_window empty;
            const auto & d = found == w.devices.end() ? empty : found->second;
            const auto k = intersect(unite(d.kernel),covered), t = intersect(unite(d.transfer),covered);
            const auto mv = intersect(unite(d.matvec), covered);
            const auto mm = intersect(unite(d.bulk_matmul), covered);
            const auto rs = intersect(unite(d.recurrent), covered);
            const auto overlap_matrix = intersect(mv, mm);
            auto combined = k; combined.insert(combined.end(), t.begin(), t.end());
            const auto both = unite(std::move(combined)); all_kernels[id] = k;
            if (!first_device) { out << ','; } first_device = false;
            out << "{\"id\":" << id << ",\"kernel_ns\":" << length(k)
                << ",\"transfer_ns\":" << length(t) << ",\"work_ns\":" << length(both)
                << ",\"kernel_parts\":" << d.kernel_parts << ",\"transfer_parts\":" << d.transfer_parts
                << ",\"matvec_ns\":" << length(mv) << ",\"bulk_matmul_ns\":" << length(mm)
                << ",\"recurrent_ns\":" << length(rs) << ",\"matvec_bulk_overlap_ns\":" << length(overlap_matrix)
                << ",\"operation_elapsed_ns\":[";
            for (size_t op=0; op<operation_count; ++op) { if(op)out<<',';out<<d.operation_ns[op]; }
            out << "],\"operation_parts\":[";
            for (size_t op=0; op<operation_count; ++op) { if(op)out<<',';out<<d.operation_parts[op]; }
            out << "],\"modes\":[";
            std::map<uint64_t, std::vector<uint64_t>> totals;
            for (const auto & s : pieces) {
                auto & v = totals[s.mode]; if (v.empty()) { v.resize(8); }
                v[0] += s.end-s.start; v[1] += overlap(k,s.start,s.end);
                v[2] += overlap(t,s.start,s.end); v[3] += overlap(both,s.start,s.end);
                v[4] += overlap(mv,s.start,s.end); v[5] += overlap(mm,s.start,s.end);
                v[6] += overlap(rs,s.start,s.end); v[7] += overlap(overlap_matrix,s.start,s.end);
            }
            bool first_mode = true;
            for (const auto & m : totals) {
                if (!first_mode) { out << ','; } first_mode = false;
                out << "{\"key\":" << m.first << ",\"wall_ns\":" << m.second[0]
                    << ",\"kernel_ns\":" << m.second[1] << ",\"transfer_ns\":" << m.second[2]
                    << ",\"work_ns\":" << m.second[3]
                    << ",\"matvec_ns\":" << m.second[4] << ",\"bulk_matmul_ns\":" << m.second[5]
                    << ",\"recurrent_ns\":" << m.second[6] << ",\"matvec_bulk_overlap_ns\":" << m.second[7] << '}';
            }
            out << "]}";
        }
        // Exact cross-device concurrency histogram during source-confirmed demand.
        std::vector<std::pair<uint64_t,int>> edges;
        for (const auto & d : all_kernels) {
            for (const auto & r : d.second) { edges.emplace_back(r.start,1); edges.emplace_back(r.end,-1); }
        }
        std::sort(edges.begin(),edges.end());
        std::vector<uint64_t> histogram(known_devices.size()+1);
        size_t at = 0; int count = 0; uint64_t pos = start;
        for (const auto & e : edges) {
            if (e.first > pos) {
                uint64_t demand = 0;
                for (const auto & s : pieces) {
                    if ((s.mode & 63) && s.start < e.first && s.end > pos) {
                        demand += std::min(s.end,e.first)-std::max(s.start,pos);
                    }
                }
                if (count>=0 && size_t(count)<histogram.size()) { histogram[size_t(count)] += demand; }
            }
            count += e.second; pos=e.first; ++at;
        }
        if (pos < end) {
            uint64_t demand = 0;
            for (const auto & s : pieces) if ((s.mode & 63) && s.end>pos) { demand+=s.end-std::max(pos,s.start); }
            if (count>=0 && size_t(count)<histogram.size()) { histogram[size_t(count)] += demand; }
        }
        out << "],\"demand_kernel_concurrency_ns\":[";
        for (size_t i=0;i<histogram.size();++i) { if(i)out<<',';out<<histogram[i]; }
        out << "]}"; return out.str();
    }
    void emit(uint64_t current, bool final=false) {
        struct snapshot { uint64_t bucket; window data; std::vector<segment> pieces; intervals covered; };
        std::vector<snapshot> snapshots;std::set<uint32_t> known_devices;
        {
        std::lock_guard<std::mutex> lock(mutex);
        known_devices=devices;
        const uint64_t last=current/SECOND;
        const uint64_t until=final ? last : (last>LAG ? last-LAG : 0);
        for(uint64_t b=next_window;b<until;++b) {
            auto & w=windows[b];
            // Empty windows must also be emitted: lack of GPU work is the signal.
            if(!w.revision)w.revision=1;
        }
        for(auto & row:windows) {
            if(row.first>=until || row.second.revision==row.second.written)continue;
            const uint64_t begin=row.first*SECOND,end=begin+SECOND;
            snapshot s{row.first,row.second,{},coverage(begin,end,current)};
            for(const auto & segment:segments(begin,end))for(const auto & r:s.covered) {
                const uint64_t lo=std::max(segment.start,r.start),hi=std::min(segment.end,r.end);
                if(hi>lo)s.pieces.push_back({lo,hi,segment.mode});
            }
            row.second.written=row.second.revision;snapshots.push_back(std::move(s));
        }
        next_window=std::max(next_window,until);
        const uint64_t keep=last>KEEP ? last-KEEP:0;
        for(auto it=windows.begin();it!=windows.end() && it->first<keep;) {
            for(const auto & d:it->second.devices)pending-=d.second.kernel.size()+d.second.transfer.size();
            it=windows.erase(it);
        }
        oldest=keep;
        const uint64_t cutoff=keep*SECOND;
        auto it=states.lower_bound(cutoff);
        if(it!=states.begin()) { --it;states.erase(states.begin(),it); }
        auto cap=capture_states.lower_bound(cutoff);
        if(cap!=capture_states.begin()){--cap;capture_states.erase(capture_states.begin(),cap);}
        }
        // No CUPTI call, interval sorting, formatting or file I/O may hold the
        // source-state lock: inference only takes it to append a small mode event.
        for(const auto & s:snapshots){
            write(render(s.bucket,s.data,s.pieces,s.covered,known_devices));++windows_written;
        }
    }
    bool capture(bool on) {
        const CUpti_ActivityKind kinds[]={CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,CUPTI_ACTIVITY_KIND_MEMCPY,
            CUPTI_ACTIVITY_KIND_MEMSET,CUPTI_ACTIVITY_KIND_MEMCPY2};
        uint64_t boundary=now();
        for(auto kind:kinds) {
            write("{\"type\":\"api_enter\",\"operation\":\""+std::string(on?"enable":"disable")+
                "\",\"kind\":"+std::to_string(int(kind))+",\"wall_ns\":"+std::to_string(wall_ns())+"}");
            if((on?enable(kind):disable(kind))!=CUPTI_SUCCESS){
                ++errors;
                for(auto cleanup:kinds)disable(cleanup);
                std::lock_guard<std::mutex> lock(mutex);
                capture_states[boundary]=false;recording=false;return false;
            }
            write("{\"type\":\"api_exit\",\"kind\":"+std::to_string(int(kind))+
                ",\"wall_ns\":"+std::to_string(wall_ns())+"}");
        }
        if(on)boundary=now();
        {
            std::lock_guard<std::mutex> lock(mutex);
            capture_states[boundary]=on;
            for(auto & w:windows)if(w.first*SECOND+SECOND>=boundary-3*SECOND)++w.second.revision;
        }
        recording=on;
        write("{\"type\":\"capture\",\"time_ns\":"+std::to_string(boundary)+
            ",\"wall_ns\":"+std::to_string(wall_ns())+",\"on\":"+(on?"true":"false")+"}");
        return true;
    }
    void report_invalid() {
        std::vector<record> samples;
        {
            std::lock_guard<std::mutex> lock(mutex);
            samples.swap(invalid_examples);
        }
        for (const auto & r : samples) {
            std::ostringstream out;
            out << "{\"type\":\"invalid_record\",\"wall_ns\":" << wall_ns()
                << ",\"start\":" << r.start << ",\"end\":" << r.end
                << ",\"device\":" << r.device << ",\"normalized_kind\":" << r.kind
                << ",\"activity_kind\":" << r.activity_kind << ",\"copy_kind\":" << r.copy_kind << ",\"transfer_bytes\":" << r.transfer_bytes << "}";
            write(out.str());
        }
    }
    void loop() {
        while(!stopping) {
            // Continuous session: changing CUPTI kernel activity while cached CUDA
            // graphs remain alive is unstable in the installed12.6 SDK. The
            // independent root timer restores the unprofiled runtime at session end.
            const auto begin=steady_ns();
            write("{\"type\":\"api_enter\",\"operation\":\"flush\",\"wall_ns\":"+std::to_string(wall_ns())+"}");
            if(flush(0)!=CUPTI_SUCCESS)++errors;
            flush_ns+=steady_ns()-begin;
            const uint64_t t=now();
            const uint64_t calibrated_at=steady_ns();
            state_clock_offset=int64_t(t)-int64_t(calibrated_at);calibration_steady=calibrated_at;
            report_invalid();
            emit(t);
            std::ostringstream out;
            out<<"{\"type\":\"clock\",\"time_ns\":"<<t<<",\"wall_ns\":"<<wall_ns()
               <<",\"callback_ns\":"<<callback_ns<<",\"flush_ns\":"<<flush_ns
               <<",\"records\":"<<records<<",\"pending\":"<<pending
               <<",\"dropped_total\":"<<drop_count<<",\"invalid_total\":"<<invalid
               <<",\"errors_total\":"<<errors<<",\"late_total\":"<<late<<"}";
            write(out.str());
            std::unique_lock<std::mutex> lock(wait_mutex);
            cv.wait_for(lock,std::chrono::seconds(1),[this]{return stopping.load();});
        }
    }
};
collector * global = nullptr;
std::mutex initialize_mutex;

void CUPTIAPI requested(uint8_t ** buffer, size_t * size, size_t * max_records) noexcept {
    *buffer=nullptr;*size=0;*max_records=0;
    auto * c=global;if(!c)return;
    if(c->buffers.fetch_add(1)>=MAX_BUFFERS){--c->buffers;++c->errors;return;}
    void * p=nullptr;
    if(posix_memalign(&p,8,BUFFER)!=0){--c->buffers;++c->errors;return;}
    *buffer=static_cast<uint8_t *>(p);*size=BUFFER;
}
void CUPTIAPI completed(CUcontext context,uint32_t stream,uint8_t * buffer,size_t,size_t valid) noexcept {
    auto * c=global;
    if(!c){free(buffer);return;}
    const uint64_t begin=steady_ns();
    try {
        std::vector<record> rows;rows.reserve(valid/256+1);
        CUpti_Activity * r=nullptr;CUptiResult result=CUPTI_SUCCESS;
        while((result=c->next(buffer,valid,&r))==CUPTI_SUCCESS){
            if(r->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
                const auto * k=reinterpret_cast<const CUpti_ActivityKernel9 *>(r);
                rows.push_back({k->start,k->end,k->deviceId,0,uint32_t(r->kind),0,operation_group(k->name),0});
            }else if(r->kind==CUPTI_ACTIVITY_KIND_MEMCPY){
                const auto * k=reinterpret_cast<const CUpti_ActivityMemcpy5 *>(r);
                rows.push_back({k->start,k->end,k->deviceId,1,uint32_t(r->kind),uint32_t(k->copyKind),0,k->bytes});
            }else if(r->kind==CUPTI_ACTIVITY_KIND_MEMSET){
                const auto * k=reinterpret_cast<const CUpti_ActivityMemset4 *>(r);
                rows.push_back({k->start,k->end,k->deviceId,2,uint32_t(r->kind),0,0,k->bytes});
            }else if(r->kind==CUPTI_ACTIVITY_KIND_MEMCPY2){
                const auto * k=reinterpret_cast<const CUpti_ActivityMemcpyPtoP4 *>(r);
                rows.push_back({k->start,k->end,k->deviceId,1,uint32_t(r->kind),uint32_t(k->copyKind),0,k->bytes});
            }
        }
        if(result!=CUPTI_ERROR_MAX_LIMIT_REACHED)++c->errors;
        size_t lost=0;
        if(c->dropped(context,stream,&lost)==CUPTI_SUCCESS)c->drop_count+=lost;
        else ++c->errors;
        c->append(rows);
    } catch(...) { ++c->errors; }
    free(buffer);--c->buffers;c->callback_ns+=steady_ns()-begin;
}
template<typename T> bool symbol(void * library,const char * name,T & out){
    out=reinterpret_cast<T>(dlsym(library,name));return out!=nullptr;
}
}

void initialize() {
    std::lock_guard<std::mutex> guard(initialize_mutex);
    if(global)return;
    const char * setting=std::getenv("LLAMA_GPU_ACTIVITY_PATH");
    if(!setting || !*setting)return;
    auto * c=new collector();global=c;
    if(const char * value=std::getenv("LLAMA_GPU_ACTIVITY_DUTY")) {
        if(strcmp(value,"continuous")!=0){++c->errors;return;}
    }
    std::string path=setting;
    const auto pos=path.find("%p");if(pos!=std::string::npos)path.replace(pos,2,std::to_string(getpid()));
    c->fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
    if(c->fd<0){++c->errors;return;}
    c->library=dlopen("/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12",RTLD_NOW|RTLD_LOCAL);
    if(!c->library || !symbol(c->library,"cuptiGetTimestamp",c->timestamp) ||
        !symbol(c->library,"cuptiActivityRegisterCallbacks",c->callbacks) ||
        !symbol(c->library,"cuptiActivityEnable",c->enable) ||
        !symbol(c->library,"cuptiActivityDisable",c->disable) ||
        !symbol(c->library,"cuptiActivityGetNextRecord",c->next) ||
        !symbol(c->library,"cuptiActivityGetNumDroppedRecords",c->dropped) ||
        !symbol(c->library,"cuptiActivityFlushAll",c->flush)){++c->errors;return;}
    if(c->callbacks(requested,completed)!=CUPTI_SUCCESS){++c->errors;return;}
    const CUpti_ActivityKind kinds[]={CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,CUPTI_ACTIVITY_KIND_MEMCPY,CUPTI_ACTIVITY_KIND_MEMSET,CUPTI_ACTIVITY_KIND_MEMCPY2};
    for(auto kind:kinds)if(c->enable(kind)!=CUPTI_SUCCESS){
        ++c->errors;
        for(auto enabled_kind:kinds)c->disable(enabled_kind);
        return;
    }
    c->started=c->now();c->next_window=c->started/SECOND;c->oldest=c->next_window;
    c->state_clock_offset=int64_t(c->started)-int64_t(steady_ns());c->calibration_steady=steady_ns();
    c->states[c->started]=0;
    c->capture_states[c->started]=true;c->recording=true;
    c->write("{\"type\":\"start\",\"schema\":1,\"pid\":"+std::to_string(getpid())+
        ",\"time_ns\":"+std::to_string(c->started)+",\"wall_ns\":"+std::to_string(wall_ns())+
        ",\"window_ns\":1000000000,\"retention_seconds\":60,\"lag_seconds\":5,\"transition_guard_ns\":2000000000,"+
        "\"mode_bits\":\"active6,prefill6,mtp6,vip6\",\"names_and_content_saved\":false,"+
        "\"operation_labels\":[\"other\",\"matvec\",\"quant_matmul\",\"gemm\",\"attention\",\"recurrent_or_triangular\",\"norm\",\"movement\",\"elementwise\",\"quantization\"],"+
        "\"operation_grouping\":\"static kernel-name heuristics; no names saved\",\"operation_elapsed_is_sum\":true,"+
        "\"operation_elapsed_scope\":\"whole second; use only fully covered windows; sums can overlap\"}");
    c->running=true;
    try {
        c->worker=std::thread([c]{
            try { c->loop(); } catch(...) { ++c->errors; }
        });
    } catch(...) {
        ++c->errors;c->running=false;
        for(auto kind:kinds)c->disable(kind);
    }
}
bool enabled(){return global && global->running.load();}
void state(uint32_t active,uint32_t prefill,uint32_t mtp,uint32_t vip){
    auto * c=global;if(!c || !c->running)return;
    const uint64_t key=std::min(active,63U)|(uint64_t(std::min(prefill,63U))<<6)|
        (uint64_t(std::min(mtp,63U))<<12)|(uint64_t(std::min(vip,63U))<<18);
    if(c->mode.exchange(key)==key)return;
    // The scheduler must never enter CUPTI/driver code. Calibrate its monotonic
    // CPU clock on the collector thread and append only a small source event.
    const uint64_t begin=steady_ns();
    const uint64_t t=uint64_t(int64_t(begin)+c->state_clock_offset.load());
    std::lock_guard<std::mutex> lock(c->mutex);
    const uint64_t waited=steady_ns()-begin;c->state_wait_ns+=waited;
    auto previous=c->state_wait_max_ns.load();
    while(previous<waited && !c->state_wait_max_ns.compare_exchange_weak(previous,waited)){}
    c->states[t]=key;
}
std::string status(){
    auto * c=global;if(!c)return "{\"enabled\":false,\"configured\":false}";
    std::ostringstream s;
    s<<"{\"configured\":true,\"enabled\":"<<(c->running?"true":"false")
      <<",\"recording\":"<<(c->recording?"true":"false")<<",\"continuous\":true,\"external_restore_required\":true"
      <<",\"duty_on_seconds\":"<<c->duty_on<<",\"duty_period_seconds\":"<<c->duty_period
      <<",\"until_wall_ns\":"<<c->until_wall_ns
      <<",\"records\":"<<c->records<<",\"dropped\":"<<c->drop_count
      <<",\"invalid\":"<<c->invalid<<",\"errors\":"<<c->errors<<",\"late\":"<<c->late
      <<",\"callback_ns\":"<<c->callback_ns<<",\"flush_ns\":"<<c->flush_ns
      <<",\"state_wait_ns\":"<<c->state_wait_ns<<",\"state_wait_max_ns\":"<<c->state_wait_max_ns
      <<",\"calibration_age_ns\":"<<(steady_ns()-c->calibration_steady.load())
      <<",\"bytes_written\":"<<c->bytes<<",\"windows_written\":"<<c->windows_written
      <<",\"pending_records\":"<<c->pending<<",\"pending_record_limit\":"<<MAX_RECORDS<<"}";return s.str();
}
void shutdown(){
    auto * c=global;if(!c)return;
    c->running=false;
    c->stopping=true;c->cv.notify_all();if(c->worker.joinable())c->worker.join();
    if(c->recording.exchange(false)) {
        for(auto kind:{CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,CUPTI_ACTIVITY_KIND_MEMCPY,CUPTI_ACTIVITY_KIND_MEMSET,CUPTI_ACTIVITY_KIND_MEMCPY2})
            if(c->disable(kind)!=CUPTI_SUCCESS)++c->errors;
    }
    if(c->flush && c->flush(0)!=CUPTI_SUCCESS)++c->errors;
    if(c->timestamp){c->report_invalid();c->emit(c->now(),true);c->write("{\"type\":\"stop\",\"status\":"+status()+"}");}
    if(c->fd>=0){close(c->fd);c->fd=-1;}
}
}
