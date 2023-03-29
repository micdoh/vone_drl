install_cu111:
	poetry install
	poetry run pip install torch==1.11.1+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html