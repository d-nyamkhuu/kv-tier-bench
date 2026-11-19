docker pull -q docker.io/vllm/vllm-openai:v0.34.0 &
ssh -o BatchMode=yes kvtb@10.88.0.63 docker pull -q docker.io/vllm/vllm-openai:v0.34.0 &
wait
echo PULL_DONE
