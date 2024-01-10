# Self Supervised Vision for Climate Downscaling

#### This is the Pytorch implementation "Self Supervised Vision for Climate Downscaling".

# Abstract

Climate change is one of the most critical challenges that our planet is facing today. Rising global temperatures are already bringing noticeable changes to Earth's weather and climate patterns with increased frequency of unpredictable and extreme weather events. Future projections for climate change research are largely based on Earth System Models (ESM), which are computer models that simulate the Earth's climate system as a whole. ESM models are a framework to integrate various physical systems and their output is bound by the enormous computational resources required for running and archiving higher-resolution simulations. In a given budget of the resources, the ESM are generally run on a coarser grid, followed by a computationally lighter downscaling process to obtain a finer resolution output. In this work, we present a deep-learning model for downscaling ESM simulation data that does not require high-resolution ground truth data for model optimization. This is realized by leveraging salient data-distribution patterns and hidden dependencies between the weather variables for an individual data point at runtime. Extensive evaluation on 2x, 3x, and 4x scaling factors demonstrates that the proposed model consistently obtains superior performance over various baselines. Improved downscaling performance and no dependence on high-resolution ground truth data make the proposed method a valuable tool for climate research and mark it as a promising direction for future research.

## Model Architecture
<p align="center">
<img width="766" alt="kdd2023model_ssl_arxiv" src="https://github.com/k-s-b/climate_sd/assets/62580782/942bb533-8b67-4c85-8428-9b4975bb87e8">
</p>




## Usage
To run model change data paths for `HRClimateDataset`, `PretrainDataset`, `NoisyDataGenerator`

## Data
High resolution CESM data can be downloaded from Ultra-high-resolution climate simulation project website [at this link](http://climatedata.ibs.re.kr/data/cesm-hires)

## Sample outputs


Paleoclimate LR data with no ground truth             |  CESM data with HR ground truth
:-------------------------:|:-------------------------:
![paleo artifacts small](https://github.com/k-s-b/self-scd/assets/62580782/c2e1f46b-8fb9-412e-8d5d-24eda970b878)  |  ![kdd artifacts small](https://github.com/k-s-b/self-scd/assets/62580782/7823bd97-53e4-40fc-9b38-12eff0a37c74)





