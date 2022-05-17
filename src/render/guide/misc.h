inline float logistic(float x) {
	return 1 / (1 + std::exp(-x));
}

//Implements the stochastic-gradient-based Adam optimizer [Kingma and Ba 2014]
class AdamOptimizer {
public:
	AdamOptimizer(float learningRate, int batchSize = 1, float epsilon = 1e-08f, float beta1 = 0.9f, float beta2 = 0.999f) {
		m_hparams = { learningRate, batchSize, epsilon, beta1, beta2 };
	}

	AdamOptimizer& operator=(const AdamOptimizer& arg) {
		m_state = arg.m_state;
		m_hparams = arg.m_hparams;
		return *this;
	}

	AdamOptimizer(const AdamOptimizer& arg) {
		*this = arg;
	}

	void append(float gradient, float statisticalWeight) {
		m_state.batchGradient += gradient * statisticalWeight;
		m_state.batchAccumulation += statisticalWeight;

		if (m_state.batchAccumulation > m_hparams.batchSize) {
			step(m_state.batchGradient / m_state.batchAccumulation);

			m_state.batchGradient = 0;
			m_state.batchAccumulation = 0;
		}
	}

	void step(float gradient) {
		++m_state.iter;

		float actualLearningRate = m_hparams.learningRate * std::sqrt(1 - pow(m_hparams.beta2, m_state.iter)) / (1 - pow(m_hparams.beta1, m_state.iter));
		m_state.firstMoment = m_hparams.beta1 * m_state.firstMoment + (1 - m_hparams.beta1) * gradient;
		m_state.secondMoment = m_hparams.beta2 * m_state.secondMoment + (1 - m_hparams.beta2) * gradient * gradient;
		m_state.variable -= actualLearningRate * m_state.firstMoment / (std::sqrt(m_state.secondMoment) + m_hparams.epsilon);

		// Clamp the variable to the range [-20, 20] as a safeguard to avoid numerical instability:
		// since the sigmoid involves the exponential of the variable, value of -20 or 20 already yield
		// in *extremely* small and large results that are pretty much never necessary in practice.
		m_state.variable = clamp(m_state.variable, -20.0f, 20.0f);
	}

	float variable() const {
		return m_state.variable;
	}

private:
	struct State {
		int iter = 0;
		float firstMoment = 0;
		float secondMoment = 0;
		float variable = 0;

		float batchAccumulation = 0;
		float batchGradient = 0;
	} m_state;

	struct Hyperparameters {
		float learningRate;
		int batchSize;
		float epsilon;
		float beta1;
		float beta2;
	} m_hparams;
};