import os
import tensorflow as tf
num_classes=10
num_examples_epoch_for_train=50000
num_examples_epoch_for_eval=10000
class CIFAR10Record(object):
    pass
def read_cifar10(file_queue):
    result=CIFAR10Record()
    labels_bytes=1
    result.height=32
    result.width = 32
    result.depth = 3
    image_bytes=result.height*result.width*result.depth
    reocrd_bytes=image_bytes+labels_bytes
    reader=tf.FixedLengthRecordReader(record_bytes=reocrd_bytes)
    result.key,value=reader.read(file_queue)
    record_bytes=tf.decode_raw(value.tf.uint8)
    result.label=tf.cast(tf.strided_slice(reocrd_bytes,[0],[labels_bytes]),tf.int32)
    depth_major=tf.reshape(tf.strided_slice(reocrd_bytes,[labels_bytes],[labels_bytes+image_bytes]),[result.depth,result.height,result.width])
    result.uint8image=tf.transpose(depth_major,[1,2,0])
    return result
def inputs(data_dir,batch_size,distorted):
    filenames=[os.path.join(data_dir,"data_batch_%d"%i) for i in range(1,6)]
    file_queue=tf.train.string_input_producer(filenames)
    read_input=read_cifar10(file_queue)
    reshaped_image=tf.cast(read_input.uint8image,tf.float32)
    num_examples_per_epoch=num_examples_epoch_for_train
    if distorted !=None:
        cropped_image=tf.random_crop(reshaped_image,[24,24,3])
        flipped_image=tf.image.random_flip_left_right(cropped_image)
        adjusted_brightness=tf.image.random_brightness(flipped_image,max_delta=0.8)
        adjusted_contrast=tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)
        float_image=tf.image.per_image_standardization(adjusted_contrast)
        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])
        min_queue_examples=int(num_examples_epoch_for_eval*0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)
        images_train,label_train=tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,
                                                       num_threads=16,capacity=min_queue_examples,min_after_dequeue=min_queue_examples)
        return images_train,tf.reshape(label_train,[batch_size])
    else:
        resized_image=tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)
        float_image=tf.image.per_image_standardization(resized_image)
        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])
        min_queue_example=int(num_examples_per_epoch*0.4)
        images_test,labels_test=tf.train.batch([float_image,read_input.label],batch_size=batch_size,
                                              num_threads=16,capacity=min_queue_example+3*batch_size)
        return images_test,tf.reshape(labels_test,[batch_size])






