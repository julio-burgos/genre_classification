
import * as tf from "@tensorflow/tfjs"

const sampleRate = 22050;
const numSeconds = 30
const fftLength = 2048;
const hopShift = 1024;
const nmels = 128;
const samplesize = numSeconds * sampleRate;
const windowSize = fftLength;

export async function getmonofromfile(event) {

    const input = event.target;
    const reader = new FileReader();
    const file = input.files[0] as File
    const arrayBuffer = await new Promise((resolve, reject) => {
        reader.onerror = () => {
            reader.abort();
            reject(new DOMException("Problem parsing input file."));
        };

        reader.onload = (ev) => resolve((ev.target as FileReader).result);;
        reader.readAsArrayBuffer(file);
    });

    const audioContext = new AudioContext({ sampleRate });
    const decoded = await audioContext.decodeAudioData(arrayBuffer as ArrayBuffer)
    const L = tf.tensor1d(decoded.getChannelData(0).slice(0, samplesize))
    const R = tf.tensor1d(decoded.getChannelData(1).slice(0, samplesize))
    const mono = L.addStrict(R).div(2).as1D()
    audioContext.close();
    return mono;
}

export async function getmelspectrogram(event) {

    const label = "getmelspectogram";
    const mono = await getmonofromfile(event)

    const xx = padding(mono, fftLength / 2)

    const S = tf.abs(tf.stft(xx, windowSize, hopShift, fftLength, tf.hannWindow)).transpose().pow(2)

    const mels = mel(sampleRate, fftLength, nmels)

    return tf.dot(mels, S)

};






function padding(x: tf.Tensor1D, pad_width: number) {

    const left = (x.slice(1, pad_width).reverse())
    const right = (x.slice(x.shape[0] - pad_width - 1, pad_width).reverse())
    return left.concat(x).concat(right)

}


function mel(sr: number, n_fft: number, n_mels: number) {
    const fmax = sr / 2;
    const fmin = 0.0;
    let weights = tf.zeros([n_mels, n_fft / 2 + 1], "float32")
    let weightsarr = tf.zeros([n_mels, n_fft / 2 + 1], "float32").arraySync()
    const fftfreqs = tf.linspace(0, sr / 2, 1 + n_fft / 2)
    const mel_f = mel_frequencies(n_mels + 2, fmin, fmax)
    const fdiff = mel_f.slice(1).sub(mel_f.slice(0, mel_f.size - 1))
    const ramps = tf.tensor(subtractwithBroadcasting(mel_f.arraySync(), fftfreqs.dataSync()))
    for (let i = 0; i < n_mels; i++) {
        const lower = tf.tensor(ramps.mul(-1).arraySync()[i]).div(fdiff.arraySync()[i])
        const upper = tf.tensor(ramps.arraySync()[i + 2]).div(fdiff.arraySync()[i + 1])

        weightsarr[i] = tf.maximum(0, tf.minimum(lower, upper)).arraySync()
        weights = tf.tensor(weightsarr)
        const diff = mel_f.slice(2, n_mels).sub(mel_f.slice(0, n_mels))
        let enorm = tf.tensor([2.0]).tile([n_mels]).div(diff);
        const temp = enorm.tile([1 + n_fft / 2]).reshape([n_mels, 1 + n_fft / 2])
        weights = weights.mul(temp);
    }


    return weights;
}



function mel_frequencies(n_mels: number, fmin: number, fmax: number) {

    const min_mel = 0
    const max_mel = hz_to_mel(fmax)

    const mels = tf.linspace(min_mel, max_mel, n_mels)


    return mel_to_hz(mels)

}

function hz_to_mel(frequencies: number) {
    const f_min = 0.0
    const f_sp = 200.0 / 3
    let mels = (frequencies - f_min) / f_sp
    const min_log_hz = 1000.0
    const min_log_mel = (min_log_hz - f_min) / f_sp
    const logstep = Math.log(6.4) / 27.0
    mels = min_log_mel + Math.log(frequencies / min_log_hz) / logstep

    return mels
}
function mel_to_hz(mels: tf.Tensor) {

    const f_min = tf.tensor(0.0)
    const f_sp = tf.tensor(200.0 / 3)
    const min_log_hz = tf.tensor(1000.0)
    const min_log_mel = min_log_hz.sub(f_min).div(f_sp)
    const logstep = tf.tensor(Math.log(6.4) / 27.0)

    const freqs = f_min.add(f_sp).mul(mels)

    const cond = mels.less(min_log_mel); // True,false


    return freqs.where(cond, min_log_hz.mul(tf.exp(logstep.mul(mels.sub(min_log_mel)))));


}


function subtractwithBroadcasting(arr1, arr2) {
    let result = new Array(arr1.length).fill(new Array(arr2.length).fill(0))
    result = arr1.map(val1 => arr2.map((val2) => val1 - val2))
    return result
}
