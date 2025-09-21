#ifndef ACTIVATIONS_SOFTFP_H_
#define ACTIVATIONS_SOFTFP_H_

float relu(float x);
float leaky_relu(float x, float negative_slope);
float elu(float x, float alpha);
float silu(float x);
float sigmoid(float x);
float tanh_act(float x);
float gelu(float x);
float mish(float x);
void softmax(const float* input, float* output, int size);

/* Stand-alone math ops (SoftFP-backed) so timing of a single
 * operation can be measured easily. */
float op_exp(float x);           /* expf(x)            */
float op_log(float x);           /* logf(x)            */
float op_pow(float x, float y);  /* powf(x, y)         */
float op_div(float x, float y);  /* x / y              */
float op_mul(float x, float y);  /* x * y              */
float op_add(float x, float y);  /* x + y              */

#endif /* ACTIVATIONS_SOFTFP_H_ */ 