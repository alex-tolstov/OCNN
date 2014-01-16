#ifndef TIMER_H_
#define TIMER_H_

#define NOMINMAX

#include <windows.h>

class TimerMillisPrecision {
private:
	/**
	 * ѕолученное значение времени.
	 */
	LARGE_INTEGER timer_value;
	/**
	 * «начение частоты.
	 */
	LARGE_INTEGER frequency;
	        
public:

	/**
	 * «апускает таймер.
	 */
	void start() {
		BEGIN_FUNCTION {
				check(QueryPerformanceCounter(&timer_value) == TRUE);
				check(QueryPerformanceFrequency(&frequency) == TRUE);
		} END_FUNCTION
	}
	        
	/**
	 * ¬озвращает, сколько времени прошло с момента последнего
	 * запуска.
	 *
	 * @return количество пройденного с момента запуска времени
	 * в милисекундах.
	 */
	unsigned int get_elapsed_time_ms() {
		BEGIN_FUNCTION {
			LARGE_INTEGER curr_timer_value;
			check(QueryPerformanceCounter(&curr_timer_value) == TRUE);
			return static_cast<unsigned int>( 
				static_cast<double>(curr_timer_value.QuadPart - timer_value.QuadPart) * 1000.0 / 
				static_cast<double>(frequency.QuadPart) 
			);
		} END_FUNCTION
	}
};

#endif // TIMER_H_