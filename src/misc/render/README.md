## Algorithms

I tried to implement some algorithms designed for path tracing. 
Turn the CMake option `KRR_BUILD_STARLIGHT` on if one wants to build these additional algorithm implementations. Note that these code may not be maintained as the main repository.

### Path Guiding (Currently deprecated)

This implements [Practical Path Guiding (PPG)](https://github.com/Tom94/practical-path-guiding), which is a path guiding algorithm targeted for CPU offline rendering. What I did is largely to simply move the original implementation from CPU to GPU. The performance is not quite satisfying for real-time purposes on GPUs. 

~~~json
	"params": {
		"spp_per_pass": 4,
		"max_memory": 16,
		"bsdf_fraction": 0.5,
		"distribution": "full",
		"stree_thres": 2000,
		"dtree_thres": 0.005,
		"auto_build": true,
		"mode": "offline",
		"sample_combination": "atomatic",
		"budget": {
			"type": "spp",
			"value": 1000
		}
	}
~~~

I also implemented a later [Variance-aware](https://github.com/iRath96/variance-aware-path-guiding) enhancement, which improves PPG on the theoretical side. Use the `distribution` parameter to select from the two methods (`radiance` for standard PPG, and `full` for the variance-aware version).

