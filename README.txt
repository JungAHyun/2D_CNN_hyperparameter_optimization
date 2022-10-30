◈ W&B(wandb)를 활용하여 CNN 하이퍼 파라미터 최적화 하기 ◈

▶ 실행 방법 
1. W&B(wandb) 계정 만들기 (https://wandb.ai/home)
2. VScode 터미널에 아래 명령 실행
   2-(1) pip install wandb
   2-(2) wandb login		 
   2-(3) wandb sweep [yaml 파일명] (예) wandb sweep w_config.yaml
   2-(4) wandb agent [sweep ID] (예) wandb agent ahyunjung/uncategorized/0hunkrvo

# 2-(3)에서 UnicodeDecodeError 발생 시 yaml 파일 새로 생성하고 다시 진행하기.


▶ 결과 확인 
1. 자신의 W&B 홈페이지에 접속하기 (https://wandb.ai/ahyunjung)
2. uncategorized 프로젝트에서 결과 확인하기
   - 각 run에 적절한 name 변경하기
   - 적적한 필터 사용하여 필요없는 결과 보이지 않게 설정하기


▶ name 설정 의미
#기존 논문에서 사용했던 파라미처를 기준으로 작은 값과 큰값을 지정하고 비교해보고자 했음.
> Type1  |  활성함수 비교 ['swish','relu','sigmoid', 'tanh', 'elu' ]
> Type2  |  Conv2D 필터개수 비교 [64, 128, 256, 512]
> Type3  |  Dense 유닛 개수 비교 [512, 1024, 2048, 4096]

