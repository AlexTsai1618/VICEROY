import * as tf from '@tensorflow/tfjs-core';
import { Syft } from '@openmined/syft.js';

const modelName = 'mymodel';
const modelVersion = '1.0.0';

// if the model is protected with authentication token (optional)
const authToken = '...';

const worker = new Syft({ gridUrl, verbose: true });
const job = await worker.newJob({ modelName, modelVersion, authToken });
job.request();

job.on('accepted', async ({ model, clientConfig }) => {
  const batchSize = clientConfig.batch_size;
  const lr = clientConfig.lr;

  // Load data.
  const [data, target] = LOAD_DATA();
  const batches = MAKE_BATCHES(data, target, batchSize);

  // Load model parameters.
  let modelParams = model.params.map((p) => p.clone());

  // Main training loop.
  for (let [dataBatch, targetBatch] of batches) {
    // NOTE: this is just one possible example.
    // Plan name (e.g. 'training_plan'), its input arguments and outputs depends on FL configuration and actual Plan implementation.
    let updatedModelParams = await job.plans['training_plan'].execute(
      job.worker,
      dataBatch, 
      targetBatch,
      batchSize,
      lr,
      ...modelParams
    );

    // Use updated model params in the next iteration.
    for (let i = 0; i < modelParams.length; i++) {
      modelParams[i].dispose();
      modelParams[i] = updatedModelParams[i];
    }
  }

  // Calculate & send model diff.
  const modelDiff = await model.createSerializedDiff(modelParams);
  await job.report(modelDiff);
});

job.on('rejected', ({ timeout }) => {
  // Handle the job rejection, e.g. re-try after timeout.
});

job.on('error', (err) => {
  // Handle errors.
}); 