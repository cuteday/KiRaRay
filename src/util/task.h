#pragma once
#include "common.h"
#include "window.h"
#include "host/timer.h"

KRR_NAMESPACE_BEGIN

enum class BudgetType { None = 0, Spp, Time }; /* For benchmarking. */

KRR_ENUM_DEFINE(BudgetType, { 
	{ BudgetType::None, "none" },
	{ BudgetType::Spp, "spp" },
	{ BudgetType::Time, "time" } })

struct Budget {
	BudgetType type;
	size_t value;
};


class RenderTask {
public:
	RenderTask() = default;
	~RenderTask() = default;
	RenderTask(Budget budget) : m_budget(budget) {}

	void reset() { 
		m_start_time = CpuTimer::getCurrentTimePoint();
		m_spp		 = 0;
	}

	float tickFrame() {
		switch (m_budget.type) {
			case BudgetType::Spp:
				m_spp++;
		}
		return getProgress();
	}

	float getElapsedTime() const {
		return CpuTimer::calcDuration(m_start_time, CpuTimer::getCurrentTimePoint()) * 1e-3;
	}

	size_t getCurrentSpp() const { return m_spp; }

	float getProgress() const {
		switch (m_budget.type) {
			case BudgetType::Spp:
				return (float) m_spp / (float) m_budget.value;
			case BudgetType::Time:
				return getElapsedTime() / m_budget.value;
			default:
				return 0.f;
		}
	}

	BudgetType getBudgetType() const { return m_budget.type; }

	Budget getBudget() const { return m_budget; }
	
	bool isFinished() const { return getProgress() >= 1.f; }

	void renderUI() {
		static const char *budgetNames[] = { "None", "Spp", "Time" };
		if (m_budget.type != BudgetType::None) {
			float progress = getProgress();
			ui::Text("Type: %s", budgetNames[(int) m_budget.type]);
			ui::Text("Progress: %.2f%% (%d/%zd)", progress * 100.f,
					 (int) (m_budget.value * progress), m_budget.value);
			if (ui::Button("reset"))
				reset();
		}
	}

private:
	Budget m_budget{};
	CpuTimer::TimePoint m_start_time{};
	size_t m_spp{ 0 };

	friend void to_json(json &j, const RenderTask &p) {
		j.update({
			{"type", p.m_budget.type},
			{"value", p.m_budget.value}
		});
	}
	friend void from_json(const json &j, RenderTask &p) {
		p.m_budget.type = j.value("type", BudgetType::None);
		p.m_budget.value = j.value("value", 0);
	}
};

KRR_NAMESPACE_END