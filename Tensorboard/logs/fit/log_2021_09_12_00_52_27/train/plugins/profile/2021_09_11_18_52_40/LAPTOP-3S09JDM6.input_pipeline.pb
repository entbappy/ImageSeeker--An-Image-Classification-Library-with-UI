	r??7??@r??7??@!r??7??@	1??z??T?1??z??T?!1??z??T?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$r??7??@??镲??A??|???@Yq?-???*	ffffXz%A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?Pk????@!?H؆??X@)?Pk????@1?H؆??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchtF??_??!b?c???k?)tF??_??1b?c???k?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism㥛? ???!???)%>u?)?
F%u??1?ewƴ?]?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapx$(???@!?????X@)-C??6j?1??9aM?=?:Preprocessing2F
Iterator::ModelM?J???!?????v?)??_?Le?1? ???58?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no91??z??T?I,?x=??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??镲????镲??!??镲??      ??!       "      ??!       *      ??!       2	??|???@??|???@!??|???@:      ??!       B      ??!       J	q?-???q?-???!q?-???R      ??!       Z	q?-???q?-???!q?-???b      ??!       JCPU_ONLYY1??z??T?b q,?x=??X@