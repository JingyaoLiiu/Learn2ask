#cd ./eval

model=/public/home/ldk/users/ljy/learn2ask/verl_sft/trained_models/llama3.2-3b-instruct-with-hint/global_step_91

# First. use terminal execute `bash model_server.sh $model`


# Second. run test.py.
# Args:
# --model: model path
# --think: If `think` in the first round
# --noise: If test on noised data
# --port: port of model server, default is 8099
python test.py --model $model --port 8989 