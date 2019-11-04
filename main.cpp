#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include "model.h"

#include "Utils/utils.h"

//static const char* input_imagePath = "images/ship.txt";

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> v;
  unsigned int i = 0;
  unsigned int j = s.find(c);
  while (j < s.size() ){
    v.push_back(s.substr(i, j - i));
    i = ++j;
    j = s.find(c, j);
    if (j >= s.size() ) {
      v.push_back(s.substr(i,s.size() ));
      break;
    }
  }
  return v;
}
int main()
{
//std::fstream in;
//std::ifstream in("combined.txt");
  //in.open("combined.txt");
  std::string linc1;
	float* hInputImage;
 float *w[NUMBER_WEIGHTS];
    /*for(int i=0;i<NUMBER_WEIGHTS;i++){
      w[i] = new float [weights[i]];	
  }

  std::cout<<"Loading weights into DDR memory"<<std::endl;
  std::cout<<"Initializing weight buffers for  each layers"<<std::endl;
  
  for (int i=0;i<NUMBER_WEIGHTS;i++)
    { 
      in.read((char *)w[i],sizeof(float)*weights[i]);
    }
    
    */
    
     std::cout<<"Files Reading Start..";
   //std::cout<<*w[0]<<"\n";
  std::fstream conv_1;
  std::fstream conv_2;
  std::fstream dense_3;
  std::fstream dense_4;
  std::fstream dense_5;
  
  //std::fstream input_label;
 conv_1.open("conv2d_124.txt");
  conv_2.open("conv2d_125.txt");
  dense_3.open("dense_29.txt");
  dense_4.open("dense_30.txt");
  dense_5.open("dense_31.txt");
  
 std::cout<<"Files Reading..";
	/* ------------------------------------- OpenCL Setup Code ------------------------------------- */
	
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
  //std::cout<<"\n Done1\n";
  //printf("%s",platforms);
	std::vector<cl::Device> devices;
	platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
 //std::cout<<devices[0];
//printf("%s",devices);

	cl::Context context(devices);

	cl::CommandQueue queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
  //std::cout<<"\n Done2\n";
	float *output_buffer = new float [OUTPUT_BUFFER_SIZE];
	float *input_buffer = new float [INPUT_BUFFER_SIZE];
    for (int i =0;i<OUTPUT_BUFFER_SIZE;i++)
    {
    	output_buffer[i] = 0;
    }
    //std::cout<<"\n Done3\n";
    
    
    std::fstream inputi;
    std::fstream input_label;
    input_label.open("test_data_labels_1.txt");
    inputi.open("testcase_store_1.dat");

    getline(conv_1 , linc1 );
    std::vector<std::string> listfilemax1 = split(linc1,' ');
    float *weight1 = new float [weights[0]];
    for (int l=0; l<weights[0]; l++)
        {weight1[l]= atof(listfilemax1.at(l).c_str());
        //std::cout<<weight1[l]<<" ";
        }
  
  getline(conv_1 , linc1 );
  std::vector<std::string> listfilemax2 = split(linc1,' ');
    float *bias1 = new float [weights[1]];
    for (int l=0; l<weights[1]; l++)
        {bias1[l]= atof(listfilemax2.at(l).c_str());
        //std::cout<<bias1[l]<<" ";
        }
        getline(conv_2 , linc1 );
  std::vector<std::string> listfilemax3 = split(linc1,' ');
    
    float *weight2 = new float [weights[2]];
    for (int l=0; l<weights[2]; l++)
        {weight2[l]= atof(listfilemax3.at(l).c_str());
        //std::cout<<weight2[l]<<" ";
        }
  getline(conv_2 , linc1 );
  std::vector<std::string> listfilemax4 = split(linc1,' ');
 // std::cout<<"\n";
    float *bias2 = new float [weights[3]];
    for (int l=0; l<weights[3]; l++)
        {bias2[l]= atof(listfilemax4.at(l).c_str());
        //std::cout<<bias2[l]<<" ";
        }
 getline(dense_3 , linc1 );
  std::vector<std::string> listfilemax5 = split(linc1,' ');
  
    float *weight3 = new float [weights[4]];
    for (int l=0; l<weights[4]; l++)
        {weight3[l]= atof(listfilemax5.at(l).c_str());
        //std::cout<<weight3[l]<<" ";
        }
 getline(dense_3 , linc1 );
  std::vector<std::string> listfilemax6 = split(linc1,' ');
 
    float *bias3 = new float [weights[5]];
    for (int l=0; l<weights[5]; l++)
        {bias3[l]= atof(listfilemax6.at(l).c_str());
        //std::cout<<bias3[l]<<" ";
        }
 getline(dense_4 , linc1 );
  std::vector<std::string> listfilemax7 = split(linc1,' ');
 
    float *weight4 = new float [weights[6]];
    for (int l=0; l<weights[6]; l++)
        {weight4[l]= atof(listfilemax7.at(l).c_str());
        //std::cout<<weight4[l]<<" ";
        }
    getline(dense_4 , linc1 );
  std::vector<std::string> listfilemax8 = split(linc1,' ');
 
    float *bias4 = new float [weights[7]];
    for (int l=0; l<weights[7]; l++)
        {bias4[l]= atof(listfilemax8.at(l).c_str());
        //std::cout<<bias4[l]<<" ";
        }
    getline(dense_5 , linc1 );
  std::vector<std::string> listfilemax9 = split(linc1,' ');
 
    float *weight5 = new float [weights[8]];
    for (int l=0; l<weights[8]; l++)
        {weight5[l]= atof(listfilemax9.at(l).c_str());
        //std::cout<<weight5[l]<<" ";
        }
    getline(dense_5 , linc1 );
  std::vector<std::string> listfilemax10 = split(linc1,' ');
 
    float *bias5 = new float [weights[9]];
    for (int l=0; l<weights[9]; l++)
        {bias5[l]= atof(listfilemax10.at(l).c_str());
        //std::cout<<bias5[l]<<" ";
        }
//std::cout<<"\n Done5\n";


    std::string lincA;
    std::string lincB;
     int count = 0;
     int testcases = 1000;
    for (int i = 0;i<testcases;i++)
    {
    getline(inputi, lincA);
    getline(input_label,lincB);
    std::vector<std::string> listfilemax = split(lincA,',');
    //std::cout<<listfilemax.size()<<"\n";
    //std::cout<<listfilemax[0]<<"\n";
    int test_label = atof(lincB.c_str());
    //std::cout<< test_label<<"\n";
    //std::cout<<"\n Done4\n";
    for (int l=0; l<(32*32); l++)
        {//std::cout<<listfilemax.at(l).c_str();
            input_buffer[l]= atof(listfilemax.at(l).c_str());
        }
        
        
        
  
   
 int weight_count = 0;
 int temp = NUMBER_LAYERS;
 temp = 7;
 for (int j =0;j<temp;j++)
 {
   if (layer[j][0]==0)
   
   {
      	
	/* ------------------------------------ Layer 1 Starts ------------------------------------ */

	int in_channels, out_channels, kernel_size;
	in_channels = layer[j][1];
	out_channels = layer[j][2];
	kernel_size = layer[j][5];
	imgRows = layer[j][4];
	imgCols = layer[j][4];		
	//std::cout<<"Performing Convolution  "<<j+1<<" "<<std::endl;
	//std::cout<<in_channels<<" "<<out_channels<<" "<<kernel_size<<" "<<imgRows;
	
  
	try
	{
		  cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_channels*imgRows*imgCols*sizeof(float));
    cl::Buffer filterBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_channels*out_channels*kernel_size*kernel_size*sizeof(float));
    cl::Buffer biasBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, out_channels*sizeof(float));
    cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, out_channels*imgRows*imgCols*sizeof(float));
    cl::Buffer in_channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer out_channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer kernelSizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer imgRowsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer imgColsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, in_channels*imgRows*imgCols*sizeof(float), input_buffer);
   //std::cout<<"Conv: weight_count = "<<weight_count<<"\n";
    switch(j){
    case 0:queue.enqueueWriteBuffer(filterBuffer, CL_TRUE, 0, in_channels*out_channels*kernel_size*kernel_size*sizeof(float), weight1);
   //std::cout<<"weight_count = "<<weight_count<<"\n";
		queue.enqueueWriteBuffer(biasBuffer, CL_TRUE, 0, out_channels*sizeof(float), bias1);
   break;
   case 2:queue.enqueueWriteBuffer(filterBuffer, CL_TRUE, 0, in_channels*out_channels*kernel_size*kernel_size*sizeof(float), weight2);
   //std::cout<<in_channels<<" "<<out_channels<<" "<<kernel_size<<" "<<imgRows;
		queue.enqueueWriteBuffer(biasBuffer, CL_TRUE, 0, out_channels*sizeof(float), bias2);
   break;
   default: std::cout<<"ERROR Conv case failure\n";
   }
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, out_channels*imgRows*imgCols*sizeof(float), output_buffer);
		queue.enqueueWriteBuffer(in_channelsBuffer, CL_TRUE, 0, sizeof(int), &in_channels);
		queue.enqueueWriteBuffer(out_channelsBuffer, CL_TRUE, 0, sizeof(int), &out_channels);
		queue.enqueueWriteBuffer(kernelSizeBuffer, CL_TRUE, 0, sizeof(int), &kernel_size);
		queue.enqueueWriteBuffer(imgRowsBuffer, CL_TRUE, 0, sizeof(int), &imgRows);
		queue.enqueueWriteBuffer(imgColsBuffer, CL_TRUE, 0, sizeof(int), &imgCols);

		std::ifstream sourceFile("cl_kernels/conv.cl");
        std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
         cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "convolution");

     	kernel.setArg(0, out_channelsBuffer);
     	kernel.setArg(1, in_channelsBuffer);
     	kernel.setArg(2, kernelSizeBuffer);
     	kernel.setArg(3, inputBuffer);
     	kernel.setArg(4, filterBuffer);
     	kernel.setArg(5, biasBuffer);
     	kernel.setArg(6, outputBuffer);
     	kernel.setArg(7, imgRowsBuffer);
     	kernel.setArg(8, imgColsBuffer);

     	cl::NDRange global(imgCols, imgRows);
     	cl::NDRange local(2, 2);
      cl::Event event;
     	queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,NULL,&event);
      queue.finish();
     	// Read data back
     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, out_channels*imgRows*imgCols*sizeof(float), output_buffer);
     cl_ulong time_start;
     cl_ulong time_end;
     
     event.wait();
    double total_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end); 
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    total_time = time_end - time_start;

/* Results */
std::cout << "Execution time in milliseconds for convolution layer " << total_time*1.0e-6f << std::endl;   
	}
 
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}
    weight_count = weight_count+2;
    std::cout<<(out_channels*imgRows*imgCols)<<"\n";
    for (int p = 0;p<(out_channels*imgRows*imgCols);p++)
          { 
               
              input_buffer[p] = output_buffer[p]; 
              //if(j==2){std::cout<<input_buffer[p]<<" ";}
      }
    }
   
    else if (layer[j][0]==1)
      {//std::cout<<"Maxpooling"<<std::endl;
	/* ------------------------------------ Layer 1 Ends ------------------------------------ */

	/*

	/* ------------------------------------ MaxPool 2D Starts ------------------------------------ */

	int channels, pool_size, outImgRows, outImgCols;
	channels = layer[j][1];
	imgRows = layer[j][3];
	imgCols = layer[j][3];
	pool_size = 2;

	outImgRows = get_post_maxPool_size(pool_size, imgRows);
	outImgCols = get_post_maxPool_size(pool_size, imgCols);
	for (int i =0;i<channels*outImgCols*outImgCols;i++)
 {
  output_buffer[i] = 0;
 }


	try
	{
		cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, channels*imgRows*imgCols*sizeof(float));
		cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, channels*outImgRows*outImgCols*sizeof(float));
		cl::Buffer channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer poolSizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer inDimBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer outDimBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, channels*imgRows*imgCols*sizeof(float), input_buffer);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, channels*outImgRows*outImgCols*sizeof(float), output_buffer);
		queue.enqueueWriteBuffer(channelsBuffer, CL_TRUE, 0, sizeof(int), &channels);
		queue.enqueueWriteBuffer(poolSizeBuffer, CL_TRUE, 0, sizeof(int), &pool_size);
		queue.enqueueWriteBuffer(inDimBuffer, CL_TRUE, 0, sizeof(int), &imgRows);
		queue.enqueueWriteBuffer(outDimBuffer, CL_TRUE, 0, sizeof(int), &outImgRows);

		std::ifstream sourceFile("cl_kernels/max_pool2d.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "max_pool2d");

     	kernel.setArg(0, channelsBuffer);
     	kernel.setArg(1, inDimBuffer);
     	kernel.setArg(2, poolSizeBuffer);
     	kernel.setArg(3, outDimBuffer);
     	kernel.setArg(4, inputBuffer);
     	kernel.setArg(5, outputBuffer);

     	cl::NDRange global(outImgRows, outImgCols);
     	cl::NDRange local(1, 1);
     cl::Event event;
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,NULL,&event);
      queue.finish();

     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, channels*outImgRows*outImgCols*sizeof(float), output_buffer);
         cl_ulong time_start;
     cl_ulong time_end;
     
     event.wait();
    double total_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end); 
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    total_time = time_end - time_start;

/* Results */
std::cout << "Execution time in milliseconds for maxpool layer " << total_time*1.0e-6f << std::endl;   

	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}
 std::cout<<(channels*outImgRows*outImgCols)<<"\n";
 for (int p = 0;p<(channels*outImgRows*outImgCols);p++)
          { //if(j==1){std::cout<<output_buffer[p]<<" ";}
               
              input_buffer[p] = output_buffer[p]; 
              //if(j==3){std::cout<<input_buffer[p]<<" ";}
      }
   
      
    }

     else
     {std::cout<<""<<std::endl;

	/* ------------------------------------ Fully Connected 1 Starts ------------------------------------ */
	
	int in_features, out_features;
	in_features = layer[j][1];
	out_features = layer[j][2];
  //std::cout<<in_features<<"\n";
  //std::cout<<out_features<<"\n";

	//std::cout<<"Performing Fully Connected "<<j<<" "<<std::endl;
  //std::cout<<j<<"\n";
	try
	{ //std::cout<<"Entering try\n";
		cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_features*sizeof(float));
		cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, out_features*sizeof(float));
		cl::Buffer weightsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_features*out_features*sizeof(float));
		cl::Buffer biasesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, out_features*sizeof(float));
		cl::Buffer inFeaturesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer outFeaturesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
//std::cout<<j+1<<"\n";
		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, in_features*sizeof(float), input_buffer);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, out_features*sizeof(float), output_buffer);
  // std::cout<<j+2<<"\n";
  //std::cout<<"Dense: weight_count = "<<weight_count<<"\n";
  switch(j){
  case 4:
  //std::cout<<"\n"<<"weight3"<<"\n";
		queue.enqueueWriteBuffer(weightsBuffer, CL_TRUE, 0, in_features*out_features*sizeof(float), weight3);
   //std::cout<<"weight_count = "<<weight_count<<"\n";
		queue.enqueueWriteBuffer(biasesBuffer, CL_TRUE, 0, out_features*sizeof(float), bias3);
   break;
   case 5: queue.enqueueWriteBuffer(weightsBuffer, CL_TRUE, 0, in_features*out_features*sizeof(float), weight4);
   //std::cout<<"weight_count = "<<weight_count<<"\n";
		queue.enqueueWriteBuffer(biasesBuffer, CL_TRUE, 0, out_features*sizeof(float), bias4);
   break;
   case 6: queue.enqueueWriteBuffer(weightsBuffer, CL_TRUE, 0, in_features*out_features*sizeof(float), weight5);
   //std::cout<<"weight_count = "<<weight_count<<"\n";
		queue.enqueueWriteBuffer(biasesBuffer, CL_TRUE, 0, out_features*sizeof(float), bias5);
   break;
   default: std::cout<<"ERROR Fully Connected Failure";
   }
   //std::cout<<"weight_count = "<<weight_count<<"\n";
		queue.enqueueWriteBuffer(inFeaturesBuffer, CL_TRUE, 0, sizeof(int), &in_features);
		queue.enqueueWriteBuffer(outFeaturesBuffer, CL_TRUE, 0, sizeof(int), &out_features);
//std::cout<<"enqued"<<"\n";
   std::ifstream sourceFile("cl_kernels/relu_linear.cl");
   //std::ifstream sourceFile2("cl_kernels/linear.cl");
   cl::Program program;
   //if(j!=6)
     // {
      std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));
         cl::Program program1 = cl::Program(context, source);
         std::cout<<"Relu Linear\n";
         program = program1; //}
         /*
   else
     {std::string sourceCode(std::istreambuf_iterator<char>(sourceFile2),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));
         cl::Program program1 = cl::Program(context, source);
         std::cout<<"Linear\n";
         program = program1;}*/
      
     	program.build(devices);
     	
     	cl::Kernel kernel(program, "relu_linear");

     	kernel.setArg(0, inFeaturesBuffer);
     	kernel.setArg(1, outFeaturesBuffer);
     	kernel.setArg(2, inputBuffer);
     	kernel.setArg(3, weightsBuffer);
     	kernel.setArg(4, biasesBuffer);
     	kernel.setArg(5, outputBuffer);

     	cl::NDRange global(out_features, 1);
     	cl::NDRange local(1, 1);
     cl::Event event;
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,NULL,&event);
      queue.finish();

     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, out_features*sizeof(float), output_buffer);
               cl_ulong time_start;
     cl_ulong time_end;
     
     event.wait();
    double total_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end); 
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    total_time = time_end - time_start;

/* Results */
std::cout << "Execution time in milliseconds for Fully Connected/Dense layer " << total_time*1.0e-6f << std::endl;   
//std::cout<<j<<"\n";
std::cout<<out_features<<"\n";
	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}
     for (int n = 0; n < out_features; n++) {
              input_buffer[n] =  output_buffer[n];
              //std::cout<<input_buffer[n]<<" ";
      }
    weight_count = weight_count+2;  
    } 

}  

  float max = input_buffer[0];
      int outputs = 0;
      for (int j = 0; j < 43; j++)
      { //std::cout<<" "<<input_buffer[j];

          if (input_buffer[j] > max)
         {
            outputs = j;
            max = input_buffer[j];
        }
      }

      std::cout<<i<<"   Predicted "<<outputs<<" "<<"Expected" << test_label;
      if ( outputs != test_label )
      {
          std ::cout<<"      "<<"Mismatched";

      }
         else
       {
          
              count = count+1;
        
      }
     std::cout<<" Done"<<std::endl;
    

}
//std::cout<<"count= "<<count<<"\n"<<"Test cases : "<<testcases;
std::cout<<std::endl<<"Accuracy : "<<((float)(count)/testcases)*100<<"\n"; 
  
  
	return 0;
 

}
