---
layout: post
title:  "A Concise Introduction to Fully homomorphic encryption"
# published: false
# mathjax: true
date:   2021-12-24 18:30:13 -0400
categories: default
tags: [Homomorphic Encryption, Privacy Preserving]
---

Recently, I have been working on the privacy preserving on machine learning. Fully homomorphic encryption is one of the promising ways to do privacy preserving.

##### What is Fully homomorphic encryption?

Fully homomorphic encryption (FHE) is a scheme to keep the encrypted data and operate on the encrypted data without decrypting it.  Data providers might just provide encrypted data to service providers. The service providers run their services on their cloud/platform with the encrypted data all the way along and return the results to data providers.  Then data providers decrypt the result on their sides. It is useful for privacy preserving scenario such as medical record data, personal data and other sensitive data.

It is kind of complementary to the secure multiparty computation.

Mathematically, We define a basic operation $$\boldsymbol{op}$$ standing for "+" , "-" and "*". 
Assume there exist two variables $$a$$ and $$b$$, we have,

$$Encrypt(a) \,op  \, Encrypt(b) = Encrypt(a  \,op \, b)$$

#####  Advantage: 

— The data stays encrypted at all times, which is suitable for cloud computation

— There is no need to mask or drop any features in order to preserve the privacy of data.

#####  Disadvantage:

The computation is heavy and infeaible for commercial usage.

#####  Use case:

 Although it involves heavy computation currently, this seems suitable for the machine learning inference with a pretrained machine learning model.  Also, there are some research on simple image classification with high accuracy, such as the [handwiritten recognition](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/CryptonetsTechReport.pdf), which indicate seems promising.

#####  Libraries:

There are several  open source library now. Each library has it’s own APIs.

Examples are [HElib](https://github.com/homenc/HElib),  Microsoft [SEAL](https://github.com/microsoft/SEAL)  and [Google FHE](https://github.com/google/fully-homomorphic-encryption).

#####  Test

Here I am testing the fully homomorphic encryption scheme based on Microsoft SEAL library with a python interface ([TenSEAL](https://github.com/OpenMined/TenSEAL))

After installing the tenseal library, we could use it to operate on 1D vectors, 2D tensors.

Here it shows the encryption and decryption on the vector operation. 

{% highlight python %} 
# !pip install tenseal
# define FHE context for polynomial rings parameters, public key and secret key.
def context():
    # Create TenSEAL context
    bits_scale = 26

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=2**13,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, 31]
    )
    # set the scale
    context.global_scale = pow(2, bits_scale)

    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    return context

context = context()
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
