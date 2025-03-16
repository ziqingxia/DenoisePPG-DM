stage=$1
model_name=$2
ckp=$3

. ./path.sh

voicebank_noisy="${voicebank}/noisy"
voicebank_clean="${voicebank}/clean"

wav_root=${voicebank_noisy}
spec_root=${output_path}/spec/voicebank_Noisy_Test
spec_type="noisy spectrum"


if [[ ${stage} -le 1 ]]; then
    echo "stage 1 : preparing testing data"

    spec_path=${spec_root}
    wave_path=${wav_root}
    echo "create ${spec_type} from ${wave_path} to ${spec_path}"
    rm -r ${spec_path} 2>/dev/null
    mkdir -p ${spec_path}
    python src/cdiffuse/preprocess.py ${wave_path} ${spec_path} --test --voicebank
fi

if [[ ${stage} -le 2 ]]; then
    echo "stage 2 : inference model"

    test_spec_list=${spec_root}
    
    enhanced_path=${output_path}/Test_Result/${model_name}/model${ckp}/test
    rm -r ${enhanced_path} 2>/dev/null
    mkdir -p ${enhanced_path} 
    echo "inference enhanced wav file from ${spec_root} to ${enhanced_path}"
    python src/cdiffuse/test.py ${output_path}/${model_name}/${ckp}.pt ${test_spec_list} ${voicebank_noisy} -o ${enhanced_path} --se --voicebank
fi