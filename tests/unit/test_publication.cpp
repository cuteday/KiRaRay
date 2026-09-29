#include "integrations/hydra/publication.h"

#include <iostream>
#include <stdexcept>

using namespace krr::hydra;
using namespace std::chrono_literals;

void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}

int main() {
	try {
		PublicationSchedule schedule;
		const PublicationSchedule::Clock::time_point start{};
		require(!schedule.due(0, 128, start), "An empty image must not be published");
		require(schedule.due(1, 128, start), "The first sample must publish immediately");
		schedule.submitted(1, start);
		require(!schedule.due(1, 128, start + 1ms), "An in-flight image must not be resubmitted");
		require(!schedule.due(2, 128, start + 1ms), "Pending readback must not trigger a submission every sample");
		require(schedule.publicationDue(1, 128, start), "A completed first image must publish promptly");
		schedule.published(1, start, 350.0);
		require(schedule.intervalMs() == 75.0, "First-image initialization must not throttle steady publication");
		require(!schedule.due(1, 128, start + 1s), "An unchanged image must not be republished");
		require(!schedule.due(2, 128, start + 74ms), "Intermediate updates must stay below 15 Hz");
		require(schedule.due(2, 128, start + 75ms), "A new image must publish after the interval");
		schedule.submitted(2, start + 75ms);
		schedule.published(2, start + 75ms, 20.0);
		require(schedule.intervalMs() == 200.0, "Expensive snapshots must increase the publication interval");
		require(!schedule.due(3, 128, start + 274ms), "Adaptive pacing must use elapsed time");
		require(schedule.due(3, 128, start + 275ms), "Adaptive pacing delayed an eligible image");
		require(!schedule.needsWait(5), "A bounded number of samples can remain pending");
		require(schedule.needsWait(6), "Pending GPU work must be bounded without publication");
		schedule.synchronized(6);
		require(!schedule.needsWait(6), "A completed wait must drain pending work");
		schedule.synchronized(2);
		require(!schedule.needsWait(6), "Collecting an older image must not regress completed GPU work");
		schedule.submitted(7, start + 275ms);
		schedule.published(7, start + 275ms, 200.0);
		require(schedule.intervalMs() == 500.0, "Adaptive pacing must bound preview latency");
		require(!schedule.due(8, 128, start + 774ms), "An expensive readback was not throttled");
		require(schedule.due(8, 128, start + 775ms), "Adaptive pacing exceeded its maximum interval");
		require(schedule.due(128, 128, start + 276ms), "The final sample must bypass adaptive throttling");
		schedule.submitted(8, start + 775ms);
		schedule.published(8, start + 775ms, 0.0);
		schedule.submitted(9, start + 1275ms);
		schedule.published(9, start + 1275ms, 0.0);
		require(schedule.intervalMs() == 275.0, "Publication cadence did not recover after cheaper snapshots");
		schedule.submitted(10, start + 1550ms);
		schedule.published(10, start + 1550ms, 0.0);
		schedule.submitted(11, start + 1825ms);
		schedule.published(11, start + 1825ms, 0.0);
		require(schedule.intervalMs() == 75.0, "Recovered publication cadence exceeded the frequency cap");
		schedule.submitted(12, start + 1900ms);
		schedule.submitted(13, start + 1975ms);
		schedule.published(12, start + 2000ms, 1.0);
		require(!schedule.publicationDue(13, 128, start + 2001ms), "Delayed completions must not produce publication bursts");
		require(schedule.publicationDue(13, 128, start + 2075ms), "A completed image must publish after the interval");
		require(schedule.publicationDue(128, 128, start + 2001ms), "Final completion must bypass the publication cap");
		schedule.reset();
		require(schedule.due(1, 128, start + 1826ms), "Scene changes must publish a first image promptly");
		require(schedule.intervalMs() == 75.0, "Scene changes must clear the cost history");
		require(schedule.due(1, 1, start), "A single-sample render must publish");
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
