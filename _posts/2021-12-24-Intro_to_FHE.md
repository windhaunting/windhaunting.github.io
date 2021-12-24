---
layout: post
title:  "Concise Intro to Fully homomorphic encryption"
# published: false
date:   2021-12-24 18:30:13 +0800
categories: default
tags: Encryption
---

Recently, I have working on the privacy preserving on machine learning. Fully homomorphic encryption is one of the promising magic to do privacy preserving.

##### What is Fully homomorphic encryption?

Fully homomorphic encryption (FHE) is a scheme to keep the encrypted data and operate on the encrypted data without decrypting it.  Data providers might just provide encrypted data to service providers and the service providers run their service on their cloud/platform with the encrypted data all the way along and return the result to data providers.  Then data providers decrypt the result on their side. It is useful for privacy preserving scenario such as medical  record data, personal data and other sensitive data.

It is kind of complementary to the secure multiparty computation.

Mathematically, there are basic operation op stands for "+" , "-" and "*". 
Assume there exist two variable $a$ and $b$,

$$ Encrypt(a) op Encrypt(b) = Encrypt(a op b) $$

#####  Advantage: 

— The data stays encrypted at all times; suitable for cloud computation

— is no need to mask or drop any features in order to preserve the privacy of dat

#####  Disadvantage:

Computation heavy and infeaible for commercial usage.

#####  Use case:

 Although the heavy computation, this might be suitable for inference with the pretrained machine learning model.  Also, there are some research on simple image classification with high accuracy, such as the [handwiritten recognition](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/CryptonetsTechReport.pdf), which indicate seems promising.

#####  Libraries:

There are several  open source library now. Each library has it’s own API.

Examples are [HElib](https://github.com/homenc/HElib),  Microsoft [SEAL](https://github.com/microsoft/SEAL)  and [Google FHE](https://github.com/google/fully-homomorphic-encryption)

#####  Test

Here I am testing the fully homomorphic encryption scheme based on Microsoft SEAL library with python interface ([TenSEAL](https://github.com/OpenMined/TenSEAL))

After using pip install tenseal library, we could use it operate to encrypt 1D vector, 2D tensors.


{% highlight python %} 

f1 = [2.0]
f2 = [3.0]
encr1 = ts.ckks_vector(context, f1)
encr2 = ts.ckks_vector(context, f2)

print('encr1: ', encr1)
print('encr2: ', encr2)

# operation in the cloud
test_add = encr1 + encr2
test_sub = encr1 - encr2
test_mul = encr1 * encr2


# decrypt on the client
decr_add = test_add.decrypt()
decr_sub = test_sub.decrypt()
decr_mul = test_mul.decrypt()

print('decr_add: ', decr_add)
print('decr_sub: ', decr_sub)
print('decr_mul: ', decr_mul)
{% endhighlight %}
