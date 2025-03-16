stage=$1
model_name=$2
ckp=$3

. ./path.sh

voicebank_noisy="${voicebank}/noisy"
voicebank_clean="${voicebank}/clean"



wav_root=${voicebank_noisy}
spec_root=${output_path}/spec/voicebank_Noisy/valid
spec_type="noisy spectrum"


if [[ ${stage} -le 1 ]]; then
    echo "stage 1 : inference model"
    target_wav_root=${voicebank_clean}

    test_spec_list=${spec_root}
    
    enhanced_path=${output_path}/Valid_Result/${model_name}/model${ckp}/
    rm -r ${enhanced_path} 2>/dev/null
    mkdir -p ${enhanced_path} 
    echo "inference enhanced wav file from ${spec_root} to ${enhanced_path}"
    
    python src/inference.py  ${output_path}/${model_name}/weights-${ckp}.pt ${test_spec_list} ${voicebank_noisy} -o ${enhanced_path} --se --voicebank
fi
