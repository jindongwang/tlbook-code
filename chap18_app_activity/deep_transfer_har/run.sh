for lr in 0.01 0.001 0.005
do
    for batch in 128 256 512 1024
    do
        for lamb in 0.1 0.2 0.3 0.5 0.7 0.8 1 5
        do 
            for loss in mmd coral
            do
                echo $lr $batch $lamb $loss
                python deep_transfer_har.py --lr $lr --batchsize $batch --lamb $lamb --loss $loss
            done
        done
    done
done

# # 正常加test结果：
# 0.005 256 5 mmd  -> 0.7599
# 0.01 256 5 mmd ->   0.7610

# # 正常不加test结果
# 0.01 256 5 mmd -> 0.7456

# # 用generator结果
# 0.005 128 5 mmd -> 0.7489