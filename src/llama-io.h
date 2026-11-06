#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include <utility>
#include <vector>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    virtual bool requires_tensor_data() const { return true; }
    bool has_retained_tensor_storage() const { return !tensor_storage.empty(); }
    void retain_tensor_storage(std::shared_ptr<void> owner) { tensor_storage.push_back(std::move(owner)); }

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);

private:
    // Derived writers may consume tensors in their destructor.
    std::vector<std::shared_ptr<void>> tensor_storage;
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
