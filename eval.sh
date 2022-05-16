
function evaluate() {
    train_opponent=$1
    model=$2
    eval_opponent=$3
    frame_trained=$4
    echo "\n-------------------------\n\n\n"
    echo "evaluating: (training with: $train_opponent using model: $model with $frame_trained frames) vs ($eval_opponent)"
    python evaluate.py --model_type $model --train_opponent $train_opponent --eval_opponent $eval_opponent --frames_trained $frame_trained
    echo "\n-------------------------\n"
}

echo "============================================ \n\n\n"
echo "evaluating training opponent = ppo \n\n"

# eval vs ppo
evaluate 'ppo' 'residual' 'ppo' 2006400
evaluate 'ppo' 'conv' 'ppo' 2009600
evaluate 'ppo' 'convres' 'ppo' 2009600

# eval vs prior
evaluate 'ppo' 'residual' 'prior' 2006400
evaluate 'ppo' 'conv' 'prior' 2009600
evaluate 'ppo' 'convres' 'prior' 2009600

# eval vs random
evaluate 'ppo' 'residual' 'random' 2006400
evaluate 'ppo' 'conv' 'random' 2009600
evaluate 'ppo' 'convres' 'random' 2009600

echo "============================================ \n\n\n"
echo "evaluating training opponent = prior \n\n"

# eval vs prior
evaluate 'prior' 'residual' 'ppo' 2006400
evaluate 'prior' 'conv' 'ppo' 2009600
evaluate 'prior' 'convres' 'ppo' 2009600

# eval vs prior
evaluate 'prior' 'residual' 'prior' 2006400
evaluate 'prior' 'conv' 'prior' 2009600
evaluate 'prior' 'convres' 'prior' 2009600

# eval vs random
evaluate 'prior' 'residual' 'random' 2595200
evaluate 'prior' 'conv' 'random' 2009600
evaluate 'prior' 'convres' 'random' 2009600

###############################################################################

echo "============================================ \n\n\n"
echo "evaluating training opponent = random \n\n"

evaluate 'random' 'residual' 'ppo' 4851200 # !!!!! this one seems to be the best, 70% winning rate
evaluate 'random' 'conv' 'ppo' 2009600
evaluate 'random' 'convres' 'ppo' 2009600

evaluate 'random' 'residual' 'prior' 4851200 # !!!!! this one seems to be the best, 70% winning rate
evaluate 'random' 'conv' 'prior' 2009600
evaluate 'random' 'convres' 'prior' 2009600

evaluate 'random' 'residual' 'random' 4851200 # !!!!! this one seems to be the best, 70% winning rate
evaluate 'random' 'conv' 'random' 2009600
evaluate 'random' 'convres' 'random' 2009600