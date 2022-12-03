#include "image.h"
#include "logger.h"
#include "tinyexr.h"

#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN


namespace tinyexr {
void save_exr(const float *data, int width, int height, int nChannels, int channelStride,
			  const char *outfilename, bool flip) {
	EXRHeader header;
	InitEXRHeader(&header);

	EXRImage image;
	InitEXRImage(&image);

	image.num_channels = nChannels;

	std::vector<std::vector<float>> images(nChannels);
	std::vector<float *> image_ptr(nChannels);
	for (int i = 0; i < nChannels; ++i) {
		images[i].resize((size_t) width * (size_t) height);
	}

	for (int i = 0; i < nChannels; ++i) {
		image_ptr[i] = images[nChannels - i - 1].data();
	}

	for (int c = 0; c < nChannels; c++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				// whether flip vertically?
				int ry			= flip ? height - 1 - y : y;
				int idx			= y * width + x;
				int ridx		= ry * width + x;
				images[c][ridx] = data[channelStride * idx + c];
			}
		}
	}

	image.images = (unsigned char **) image_ptr.data();
	image.width	 = width;
	image.height = height;

	header.line_order	= 1;
	header.num_channels = nChannels;
	header.channels		= (EXRChannelInfo *) malloc(sizeof(EXRChannelInfo) * header.num_channels);
	// Must be (A)BGR order, since most of EXR viewers expect this channel order.
	strncpy(header.channels[0].name, "B", 255);
	header.channels[0].name[strlen("B")] = '\0';
	if (nChannels > 1) {
		strncpy(header.channels[1].name, "G", 255);
		header.channels[1].name[strlen("G")] = '\0';
	}
	if (nChannels > 2) {
		strncpy(header.channels[2].name, "R", 255);
		header.channels[2].name[strlen("R")] = '\0';
	}
	if (nChannels > 3) {
		strncpy(header.channels[3].name, "A", 255);
		header.channels[3].name[strlen("A")] = '\0';
	}

	header.pixel_types			 = (int *) malloc(sizeof(int) * header.num_channels);
	header.requested_pixel_types = (int *) malloc(sizeof(int) * header.num_channels);
	for (int i = 0; i < header.num_channels; i++) {
		header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
		header.requested_pixel_types[i] =
			TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
	}

	const char *err = NULL; // or nullptr in C++11 or later.
	int ret			= SaveEXRImageToFile(&image, &header, outfilename, &err);
	if (ret != TINYEXR_SUCCESS) {
		std::string error_message = std::string("Failed to save EXR image: ") + err;
		FreeEXRErrorMessage(err); // free's buffer for an error message
		throw std::runtime_error(error_message);
	}
	printf("Saved exr file. [ %s ] \n", outfilename);

	free(header.channels);
	free(header.pixel_types);
	free(header.requested_pixel_types);
}
} // namespace tinyexr

namespace pfm {

static constexpr bool hostLittleEndian =
#if defined(__BYTE_ORDER__)
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	true
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
	false
#else
#error "__BYTE_ORDER__ defined but has unexpected value"
#endif
#else
#if defined(__LITTLE_ENDIAN__) || defined(__i386__) || defined(__x86_64__) || defined(_WIN32) ||   \
	defined(WIN32)
	true
#elif defined(__BIG_ENDIAN__)
	false
#elif defined(__sparc) || defined(__sparc__)
	false
#else
#error "Can't detect machine endian-ness at compile-time."
#endif
#endif
	;

static inline int isWhitespace(char c) { return c == ' ' || c == '\n' || c == '\t'; }

static int readWord(FILE *fp, char *buffer, int bufferLength) {
	int n;
	int c;

	if (bufferLength < 1)
		return -1;

	n = 0;
	c = fgetc(fp);
	while (c != EOF && !isWhitespace(c) && n < bufferLength) {
		buffer[n] = c;
		++n;
		c = fgetc(fp);
	}

	if (n < bufferLength) {
		buffer[n] = '\0';
		return n;
	}

	return -1;
}

/* out: data[rgb], res_x, res_y */
Color4f *ReadImagePFM(const std::string &filename, int *xres, int *yres) {
	constexpr int BUFFER_SIZE = 80;
	float *data				  = nullptr;
	Color4f *rgb			  = nullptr;
	char buffer[BUFFER_SIZE];
	unsigned int nFloats;
	int nChannels, width, height;
	float scale;
	bool fileLittleEndian;

	FILE *fp = fopen(filename.c_str(), "rb");
	if (!fp)
		goto fail;

	// read either "Pf" or "PF"
	if (readWord(fp, buffer, BUFFER_SIZE) == -1)
		goto fail;

	if (strcmp(buffer, "Pf") == 0)
		nChannels = 1;
	else if (strcmp(buffer, "PF") == 0)
		nChannels = 3;
	else
		goto fail;

	// read the rest of the header
	// read width
	if (readWord(fp, buffer, BUFFER_SIZE) == -1)
		goto fail;
	width = atoi(buffer);
	*xres = width;

	// read height
	if (readWord(fp, buffer, BUFFER_SIZE) == -1)
		goto fail;
	height = atoi(buffer);
	*yres  = height;

	// read scale
	if (readWord(fp, buffer, BUFFER_SIZE) == -1)
		goto fail;
	sscanf(buffer, "%f", &scale);

	// read the data
	nFloats = nChannels * width * height;
	data	= new float[nFloats];
	// Flip in Y, as P*M has the origin at the lower left.
	for (int y = height - 1; y >= 0; --y) {
		if (fread(&data[y * nChannels * width], sizeof(float), nChannels * width, fp) !=
			nChannels * width)
			goto fail;
	}

	// apply endian conversian and scale if appropriate
	fileLittleEndian = (scale < 0.f);
	if (hostLittleEndian ^ fileLittleEndian) {
		uint8_t bytes[4];
		for (unsigned int i = 0; i < nFloats; ++i) {
			memcpy(bytes, &data[i], 4);
			std::swap(bytes[0], bytes[3]);
			std::swap(bytes[1], bytes[2]);
			memcpy(&data[i], bytes, 4);
		}
	}
	if (std::abs(scale) != 1.f)
		for (unsigned int i = 0; i < nFloats; ++i)
			data[i] *= std::abs(scale);

	// create RGBs...
	rgb = new Color4f[width * height];
	if (nChannels == 1) {
		for (int i = 0; i < width * height; ++i)
			rgb[i] = Color4f(data[i]);
	} else {
		for (int i = 0; i < width * height; ++i)
			rgb[i] = { data[3 * i], data[3 * i + 1], data[3 * i + 2], 1 };
	}

	delete[] data;
	fclose(fp);
	Log(Info, "Read PFM image %s (%d x %d)", filename.c_str(), *xres, *yres);
	return rgb;

fail:
	Log(Error, "Error reading PFM file \"%s\"", filename.c_str());
	if (fp)
		fclose(fp);
	delete[] data;
	delete[] rgb;
	return nullptr;
}
} // namespace pfm

namespace image {
	

Vector2f ndir_to_oct_equal_area_unorm(Vector3f n) {
	// Use atan2 to avoid explicit div-by-zero check in atan(y/x).
	float r	  = sqrt(1.f - abs(n[2]));
	float phi = atan2(abs(n[1]), abs(n[0]));

	// Compute p = (u,v) in the first quadrant.
	Vector2f p;
	p[1] = r * phi * M_2PI;
	p[0] = r - p[1];

	// Reflect p over the diagonals, and move to the correct quadrant.
	if (n[2] < 0.f)
		p = Vector2f{ 1 - p[1], 1 - p[0] };
	p[0] = copysignf(p[0], n[0]);
	p[1] = copysignf(p[1], n[1]);
	return p * 0.5f + Vector2f(0.5f);
}
	
Color4f* convertEqualAeraOctahedralMappingToSpherical(Color4f *data, int width, int height) {
	CHECK_EQ(width, height);
	if (width != height)
		logError(
			"Converting an image that may not be an equal-area octahedral mapping environment!");
	Color4f *rgb = new Color4f[2 * width * height];
	Vector2i size{ width, height };
	Vector2i dstSize{ width * 2, height };
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width * 2; c++) {
			Vector2f pixel{ (float) c, (float) r };
			Vector2f latlong = (pixel + Vector2f(0.5f)).cwiseQuotient(Vector2f(dstSize));
			Vector3f dir	 = utils::latlongToWorld(latlong);
			Vector2i oct	 = ndir_to_oct_equal_area_unorm(dir).cwiseProduct(Vector2f(size)).cast<int>();
			oct				 = oct.cwiseMin(size - Vector2i(1)).cwiseMax(0);
			uint srcPixel	 = oct[0] + oct[1] * width;
			uint dstPixel	 = c + r * width * 2;
			rgb[dstPixel]	 = data[srcPixel];
		}
	}
	return rgb;
}

}

KRR_NAMESPACE_END

