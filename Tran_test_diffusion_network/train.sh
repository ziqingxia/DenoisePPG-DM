# export CUDA_VISIBLE_DEVICES='0,1,2'

stage=$1
model_name=$2  # e.g. "cdiffuse", "cdiffuse_pre"
pretrain_model=$3 # e.g. cdiffuse_pre/weights-ckpt.pt"
. ./path.sh

voicebank_noisy="${voicebank}/noisy"
voicebank_clean="${voicebank}/clean"

wav_root=${voicebank_noisy}  # wave语音数据 training dataset
spec_root=${output_path}/spec/voicebank_Noisy  # preprocess后的spectrum数据 training dataset
spec_type="noisy spectrum"


if [[ ${stage} -le 1 ]]; then
    echo "stage 1 : preparing training and validation data"
    wave_path=${wav_root}
    echo "create ${spec_type} from ${wave_path} to ${spec_root}/train"
    mkdir -p ${spec_root}/train
    # make training dataset wav to sepctrum
    python src/preprocess.py ${wave_path} ${spec_root}/train --voicebank
fi

if [[ ${stage} -le 2 ]]; then
    echo "stage 2 : training model"
    target_wav_root=${voicebank_clean}
    noisy_wav_root=${voicebank_noisy}

    train_spec_list=""

    spec_path=${spec_root}/train
    train_spec_list="${train_spec_list} ${spec_path}"
    
    if [ -z "$pretrain_model" ]; then
        python src/__main__.py ${output_path}/${model_name} ${target_wav_root} ${noisy_wav_root} ${train_spec_list}  --se  --voicebank
    else
        echo "Load pre-trained model"
        python src/__main__.py ${output_path}/${model_name} ${target_wav_root} ${noisy_wav_root} ${train_spec_list}  --se  --voicebank --pretrain_path ${output_path}/${model_name}/${pretrain_model}
    fi
fi