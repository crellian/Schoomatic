for entry in /home2/random_bev_carla/new/Town*/5/0; do
        echo ${entry}
        rm ${entry}/observation_rgb_84.npy
done
