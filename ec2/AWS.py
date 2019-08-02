def get_sqs(queue_name):
    global boto3
    if not boto3:
        import boto3
    sqs = boto3.resource('sqs')
    queue = sqs.get_queue_by_name(QueueName=queue_name)
    print('Queue URL:', queue.url)
    return queue
