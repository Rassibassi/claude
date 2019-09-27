import tensorflow as tf
import os

def create_reset_metric(metric, scope='reset_metrics', *metric_args, **metric_kwargs):
    """
        see https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        metric_op, update_op = metric(*metric_args, **metric_kwargs)
        vars = tf.contrib.framework.get_variables(scope, collection=tf.compat.v1.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.compat.v1.variables_initializer(vars)
    return metric_op, update_op, reset_op

def create_mean_metrics(metricsDict):
    meanMetricOpsDict = {}
    updateOps = []
    resetOps = []
    for name, tensor in metricsDict.items():
        lossOp, updateOp, resetOp = create_reset_metric(tf.compat.v1.metrics.mean, name, tensor)
        meanMetricOpsDict[name] = lossOp
        updateOps.append(updateOp)
        resetOps.append(resetOp)    
    return meanMetricOpsDict, updateOps, resetOps

def accumulatedOptimizer(optimizer, loss, trainableVariables, nMiniBatches ):
    accumulatedVars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainableVariables]
    zeroOps = [av.assign(tf.zeros_like(av)) for av in accumulatedVars]
    gradientVars = optimizer.compute_gradients(loss, trainableVariables)
    accumulateOps = [accumulatedVars[i].assign_add(gv[0]) for i, gv in enumerate(gradientVars)]
    trainStep = optimizer.apply_gradients([(accumulatedVars[i] / nMiniBatches, gv[1]) for i, gv in enumerate(gradientVars)])
    
    return trainStep, accumulateOps, zeroOps

def train(sess, optimizer, loss, metricsDict, trainingParam, feedDictFun, debug=False):

    optimizer = optimizer(learning_rate=trainingParam.learningRate)
    meanMetricOpsDict, updateOps, resetOps = create_mean_metrics(metricsDict)
    trainStep, accumulateOps, zeroOps = accumulatedOptimizer(optimizer, loss, tf.compat.v1.trainable_variables(), trainingParam.nMiniBatches)

    init = tf.compat.v1.global_variables_initializer()

    if trainingParam.summaries:        
        s = [tf.compat.v1.summary.scalar(name, metric) for name,metric in meanMetricOpsDict.items()]
        metricSummaries = tf.compat.v1.summary.merge(s)

        summaries_dir = os.path.join(trainingParam.path, 'tboard', trainingParam.summaryString)
        os.makedirs(summaries_dir, exist_ok=True)

        trainWriter = tf.compat.v1.summary.FileWriter(summaries_dir + '/train', sess.graph)
    else:
        trainWriter = None
    
    sess.run(init)

    saver = tf.compat.v1.train.Saver()
    checkpoint_path = os.path.join(trainingParam.path,'checkpoint',trainingParam.filename,'best')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    else:
        print("Restoring...", flush=True)
        saver.restore(sess=sess,save_path=checkpoint_path)

    bestLoss = 100000
    bestAcc = 0
    lastImprovement = 0

    sess.run([resetOps, zeroOps])

    for epoch in range(1, trainingParam.nEpochs+1):
        sess.run(resetOps)
        
        for batch in range(1,trainingParam.nBatches+1):
            if debug:
                print('batch: ', batch, end=' | ', flush=True)
            sess.run(zeroOps)
            if debug:
                print('miniBatch: ', end='', flush=True)
            for miniBatch in range(1,trainingParam.nMiniBatches+1):
                if debug:
                    print(miniBatch, end=', ')
                feedDict = feedDictFun(trainingParam)
                sess.run([accumulateOps, updateOps], feed_dict=feedDict)
            print('', flush=True)
            sess.run(trainStep)
                
        outMetrics = sess.run(list(meanMetricOpsDict.values()), feed_dict=feedDict)
        outMetrics = { key:val for key,val in zip(list(meanMetricOpsDict.keys()), outMetrics) }
        
        if trainingParam.summaries:
            outMetricSummaries = sess.run(metricSummaries, feed_dict=feedDict)
            trainWriter.add_summary(outMetricSummaries, epoch)
        
        earlyStoppingMetric = outMetrics[trainingParam.earlyStoppingMetric]
        if earlyStoppingMetric < bestLoss:
            bestLoss = earlyStoppingMetric
            lastImprovement = epoch
            saver.save(sess=sess, save_path=checkpoint_path)
            
        if epoch%trainingParam.displayStep == 0:
            outString = 'epoch: {:04d}'.format(epoch)
            for key, value in outMetrics.items():
                outString += ' - {}: {:.4f}'.format(key, value)
            print(outString, flush=True)

    saver.restore(sess=sess,save_path=checkpoint_path)

    sess.run(resetOps)
    for _ in range(trainingParam.evalBatches):
        feedDict = feedDictFun(trainingParam)
        sess.run(updateOps, feed_dict=feedDict)
    
    outMetrics = sess.run(list(meanMetricOpsDict.values()), feed_dict=feedDict)
    outMetrics = { key:val for key,val in zip(list(meanMetricOpsDict.keys()), outMetrics) }
    
    return sess, outMetrics