#ifndef _EV_QUEUE_H_
#define _EV_QUEUE_H_

#include <mutex>
#include <queue>
#include <condition_variable>

template <typename T>
class EvQueue
{
public:
    EvQueue(int capacity)
    {
        if (capacity >= minQueueSize && capacity <= maxQueueSize)
        {
            queueSize = capacity;
        }
        else
        {
            queueSize = defaultQueueSize;
        }
    }

    EvQueue()
    {
        queueSize = defaultQueueSize;
    }

    ~EvQueue() = default;

    bool Push(T input_value)
    {
        std::lock_guard<std::mutex> lock(m);
        if (queue_.size() < queueSize)
        {
            queue_.push(input_value);
            cond.notify_one();
            return true;
        }

        return false;
    }

    bool PopTimeout(T &out, int s)
    {
        std::unique_lock<std::mutex> lck(m);
        auto now = std::chrono::system_clock::now();
        if (cond.wait_until(lck, now + std::chrono::seconds(s), [this]()
                            { return !queue_.empty(); }))
        {
            out = queue_.front();
            queue_.pop();
            return true;
        }
        else
        {
            return false;
        }
    }

    T Pop()
    {
        std::unique_lock<std::mutex> lck(m);
        cond.wait(lck, [this]()
                  { return !queue_.empty(); });

        T tmp_ptr = queue_.front();
        queue_.pop();
        return tmp_ptr;
    }

    bool empty()
    {
        std::unique_lock<std::mutex> lck(m);
        return queue_.empty();
    }

    int size()
    {
        std::unique_lock<std::mutex> lck(m);
        return queue_.size();
    }

    void clear()
    {
        std::unique_lock<std::mutex> lck(m);
        std::queue<T> empty;
        std::swap(empty, queue_);
    }

private:
    std::queue<T> queue_;

    int queueSize;
    mutable std::mutex mutex_;
    const int minQueueSize = 1;
    const int maxQueueSize = 10000;
    const int defaultQueueSize = 10;

    mutable std::mutex m;
    std::condition_variable cond;
};

#endif
