#!/usr/bin/env python
# coding: utf-8

# In[1]:

import boto3

spots_region_name = 'us-east-1'
spots_queue_name = 'pme-spot-fleet-x1-16xlarge-fast-warm-fsx'
script = '/data/Nchen/saddle/gamma/shell.sh'
sqs = boto3.resource('sqs', region_name = spots_region_name)
queue = sqs.get_queue_by_name(QueueName = spots_queue_name)
response = queue.send_message(MessageBody = 'bash {0}'.format(script))



