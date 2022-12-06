#include <TensorFlowLite.h>

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


#include <Wire.h>
#include "Adafruit_MPR121.h"

#ifndef _BV
#define _BV(bit) (1 << (bit)) 
#endif

  int counter = 0;
  int normalizedPixels[36];
  int pixels[36][20];
  int ind = 0;

// You can have up to 4 on one i2c bus but one is enough for testing!
Adafruit_MPR121 cap = Adafruit_MPR121();

// Keeps track of the last pins touched
// so we know when buttons are 'released'
uint16_t lasttouched = 0;
uint16_t currtouched = 0;

void Set_Base_Values()
{
    if (counter<10) {
    for (uint8_t i=0; i<6; i++) {
      for (uint8_t j=0; j<6; j++) {
        int pixelVal = cap.filteredData(i)+cap.filteredData(j+6);
        normalizedPixels[i*6+j] = normalizedPixels[i*6+j]+pixelVal;
      }
    }
  }

  if (counter == 10) {
    for (uint8_t i=0; i<6; i++) {
      for (uint8_t j=0; j<6; j++) {
        normalizedPixels[i*6+j] = normalizedPixels[i*6+j]/10;
      }
    }
  }

  counter++;

  delay(10);  
}


namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  
  Serial.begin(9600);

  while (!Serial) { // needed to keep leonardo/micro from starting too fast!
    delay(10);
  }
  
  Serial.println("Adafruit MPR121 Capacitive Touch sensor test"); 
  
  // Default address is 0x5A, if tied to 3.3V its 0x5B
  // If tied to SDA its 0x5C and if SCL then 0x5D
  if (!cap.begin(0x5A)) {
    Serial.println("MPR121 not found, check wiring?");
    while (1);
  }
  Serial.println("MPR121 found!");

  for(uint8_t i=0; i<36; i++)
  {
      normalizedPixels[i] = 0;
  }

  while(counter <= 10)
    Set_Base_Values();

  pinMode(21, INPUT);    

  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // Quantize the input from floating-point to integer
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;  
  // Place the quantized input in the model's input tensor


  /***** Collect data here *****/ 
 

    // Get the currently touched pads
  currtouched = cap.touched();

  // reset our state
  lasttouched = currtouched;

    for (uint8_t i=0; i<6; i++) {
      for (uint8_t j=0; j<6; j++) {
        int pixel = cap.filteredData(i)+cap.filteredData(j+6)-normalizedPixels[i*6+j];
        pixels[i*6+j][ind] = pixel;
      }
    }

    for (uint8_t i=0; i<36; i++) {
    Serial.print(pixels[i][ind]);
    Serial.print("  ");
    }
    Serial.print("\n");
 
  if(digitalRead(21) == HIGH)
  {
    Serial.print("Input pin detected - resetting normalized values");

      for(uint8_t i=0; i<36; i++)
      {
          normalizedPixels[i] = 0;
      }

      counter = 0;
      
      while(counter <= 10)
        Set_Base_Values();
    
  }  
  // put a delay so it isn't overwhelming
  delay(10);

  /***** Inference part here *****/ 

  for(int i=0; i<36; i++)
    input->data.f[i] = pixels[i][ind];
 
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(x));
    return;
  }

  // Obtain the quantized output from model's output tensor
  int8_t y_quantized[3];
  y_quantized[0] = output->data.f[0];
  y_quantized[1] = output->data.f[1];
  y_quantized[2] = output->data.f[2];

  Serial.print("Int vals \n");
  
  for(int i=0; i<3; i++)
  {
    Serial.print(y_quantized[i]);
    Serial.print("\n");
  }

  int res = (y_quantized[0] > y_quantized[1] ? (y_quantized[0] > y_quantized[2] ? 0 : 2) : (y_quantized[1] > y_quantized[2] ? 1 : 2));

  if(res == 1)
    Serial.print("\n FIST \n");
  else if(res == 2)
    Serial.print("\n FINGER \n");
  else
    Serial.print("\n PALM \n");
    
  Serial.print("\n");
 
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
