#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "math.h"
// rm -f vid/*.png && gcc -o video video.c -lm && ./video && ffmpeg -i vid/frame-%05d.png -c:v libx264 -vf fps=30 7cycle-noise100-30fps.mp4

void writeFrame(int frame, int frameCount)
{
	int width=512;
	int height=512;
	unsigned char *image = malloc(width*height*3);

	for(int x=0; x<width; x++)
	{
		for(int y=0; y<height; y++)
		{
			// theta does one complete cycle over the frameCount
			float theta = frame/(float)frameCount * 2*M_PI;

			// 2 cycle:
			float color = sin(theta*7)/2 + .5;

			// 7 cycle:
			//float color = sin(theta*7)/2 + .5;

			// 30 cycles + weak 1 cycle 
			//float color = sin(theta*30)/4 + .5 + sin(theta)/5 + cos(theta*3)/4.5;

			
			color += (drand48()-.5)*100;  // add noise (drand48 returns 0 to 1). 
			if(color < 0) color = 0;
			if(color > 1) color = 1;

			image[(x+y*width)*3+0] = roundf(color*255);
			image[(x+y*width)*3+1] = roundf(color*255);
			image[(x+y*width)*3+2] = roundf(color*255);
		}
	}

	char filename[100];
	snprintf(filename, 100, "vid/frame-%05d.png", frame);
	stbi_write_png(filename, width, height, 3, image, width*3);

	free(image);
}

int main()
{
	int frameCount = 300;
	for(int i=0; i<frameCount; i++)
		writeFrame(i,frameCount);
}
