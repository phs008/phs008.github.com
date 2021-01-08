---
title: Tensorboard 를 활용한 embedding 방법
description: Tensorboard 를 활용한 feature 의 PCA , T-SNE 활용법
categories:
- deep learning visualizing
tags:
- tensorboard
- visualizing
toc: true
---
# Tensorboard 를 활용한 embedding 방법

Auto encoder 기반의 feature extractor 에 대한 평가를 하는데 있어서 가시화가 되었으면 하는 생각을 하게되었다.

코드적으로 각각의 feature channel 의 PCA 를 매번 처리하고 가시화 하는 방법이 있긴 하지만 매번 다른 테스트를 할때마다 PCA 코드 까지 조금씩 조금씩 변경해야 한다는 불편함이 있었다.

그렇다가 Tensorboard 로 word 를 embedding 하여 L2 또는 Cosine distance 를 보았던 기억이 떠올랐다..

가만가만 <b>요거 조금만 변경하면 feature embedding 하여 볼수 있지 않을까?</b>

바로 시도해봅시다.

## Basic of Tensorboard embedding
[여기를](https://projector.tensorflow.org/) 클릭하면 Tensorflow 를 통해 word embedding 한 케이스를 볼수 있다.

자세히 살펴보면 좌측상단에 <b>Label by</b> 라는 Label 이란 항목을 확인할수 있다.

![1](http://phs008.github.io/assets/2021-01-07/1.png)

예상해보면 Tensorboard 에 표현될 data set 은 한쌍의 data 와 label 이 되겠구나 라는 생각을 할수 있게 한다.

Tensorboard 에선 이 label 을 별도 파일로 저장한다 (metadata.tsv)

그리고 Tensorboard log 에 embedding 하고자 하는 data 를 추가하고 metadata 로 별도로 만든 label 용 파일(metadata.tsv) 을 추가하면 된다.

## Feature Embedding in Tensorboard

필자가 Tensorboard feature embedding 을 통해 보고자 했던 것은 Auto encoder 기반으로 data clustering 이 얼마만큼 잘 될수 있는지였다.

따라서 Train data set 기반으로 Auto encoder 학습을 진행하고 valid data set 과 test data set 을 통해 data clustering 이 가능한지 수행해보았다.

> 1. 일단 학습된 모델을 불러오고
>```python
>encoder_model = Auto_Encoder(args['input_img_shape'])
>encoder_model.forward_model.load_weights(save_weights_folder)
>```

> 2. Tensorboard log 에 embedding 할 데이터를 추가한다.
>```python
>test_dataset = test_dataset.batch(args['test_dataset_batch'])
>test_tqdm_iter = tqdm(test_dataset)
>embedding_values = None
>for iter in test_tqdm_iter:
>    img, gt, img_path = iter
>    # extract_layer 은 15 * 15 * channel (1024) 이다.
>    extract_layer = encoder_model.forward_model(img, training=False)
>    # 필자는 이미지 하나당 1024개의 데이터로 축소하여 embedding 하고자 했다.
>    features = tf.keras.layers.GlobalAveragePooling2D()(extract_layer)
>    # 이미지 갯수 에 따른 features 를 concatenation 하여 Test data set 에 대해 하나의 embedding_values 를 만든다.  
>    if embedding_values is None:
>        embedding_values = features
>    else:
>        embedding_values = tf.concat([embedding_values, features], 0)
>    # 각 이미지에 대한 label 을 기입한 metadata 를 생성한다. 
>    with open(os.path.join(logdir, 'metadata.tsv'), 'a+') as f:
>        for i in gt:
>            f.write("{}\n".format(i.numpy()))
>```

>3. Tensorboard 용 check point 생성
> ```python
> import tensorboard.plugins.projector as proj
> checkpoint_weight = tf.Variable(weight)
> checkpoint = tf.train.Checkpoint(embedding=checkpoint_weight)
> checkpoint.save(os.path.join(logdir, 'embedding.ckpt'))
> config = proj.ProjectorConfig()
> embedding = config.embeddings.add()
> embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
> embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
> import tensorboard.plugins as tp
> tp.projector.visualize_embeddings(logdir, config)
>```

## 결과

다음과 같은 feature embedding 에 대한 PCA 결과를 볼수있다.

![1](http://phs008.github.io/assets/2021-01-07/2.png)