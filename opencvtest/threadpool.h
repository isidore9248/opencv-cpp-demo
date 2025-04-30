#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

class threadpool
{
public:
	threadpool(int num);
	~threadpool();

	template<typename FuncType, typename ...Args>
	void enqueue(FuncType&& func, Args&&... args)
	{
		std::function<void()> task =
			std::bind(std::forward<FuncType>(func), std::forward<Args>(args)...);

		{
			std::unique_lock<std::mutex> u_lock(m_mutex);
			m_tasks.emplace(std::move(task));
		}
		m_cv.notify_one();
	}

private:
	std::vector<std::thread> m_threads;         // 线程池
	std::queue<std::function<void()>> m_tasks;  // 任务队列
	std::mutex m_mutex;                         // 互斥锁
	std::condition_variable m_cv;               // 条件变量
	bool stop;                                  // 线程池停止标志
};

//测试线程池
void threadpool_test(void);