#include "threadpool.h"
#include <iostream>
#include <chrono>

threadpool::threadpool(int num) : stop(false)
{
	for (int i = 0; i < num; i++)
	{
		m_threads.emplace_back(
			[this]
			{
				while (true)
				{
					std::unique_lock<std::mutex> u_lock(m_mutex);

					//wait函数为true时不等待
					//没有任务可处理(m_tasks 为空) 且未收到停止信号(stop 为 false) 的情况下挂起。
					m_cv.wait(u_lock,
						[this]
						{
							return !m_tasks.empty() || stop;
						}
					);

					if (stop && m_tasks.empty())
					{
						return;
					}

					std::function<void()> run_task = std::move(m_tasks.front());
					m_tasks.pop();
					u_lock.unlock();
					run_task();
				}
			}
		);
	}
}

threadpool::~threadpool()
{
	{
		std::unique_lock<std::mutex> u_lock(m_mutex);
		stop = true;
	}
	m_cv.notify_all();
	for (auto& t : m_threads)
	{
		t.join();
	}
}

// 测试线程池
void threadpool_test(void)
{
	threadpool tp(4);
	for (int i = 0; i < 10; ++i)
	{
		tp.enqueue(
			[i]
			{
				std::cout << "Task: " << i << " is running" << std::endl;
				std::this_thread::sleep_for(std::chrono::seconds(1));
				std::cout << "Task: " << i << " is done" << std::endl;
			}
		);
	}

	// 主线程等待一段时间，确保所有任务都完成
	std::this_thread::sleep_for(std::chrono::seconds(2));
}