import * as tf from '@tensorflow/tfjs';

export async function predict(x: tf.Tensor) :Promise<tf.Tensor<tf.Rank>> {

    const model = await tf.loadLayersModel(`${window.location.href}models/CNN/model/model.json`);
    return  model.predict(x) as tf.Tensor<tf.Rank> ;

}