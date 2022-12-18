#include "image.h"
#include "logger.h"
#include "tinyexr.h"

#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN

namespace tinyexr {
	
struct bfloat16 {
unsigned short int data;
public:
	bfloat16() { data = 0; }
	operator float() {
		unsigned int proc = data << 16;
		return *reinterpret_cast<float *>(&proc);
	}
	bfloat16 &operator=(float float_val) {
		data = (*reinterpret_cast<unsigned int *>(&float_val)) >> 16;
		return *this;
	}
};
	
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

	header.line_order	= 0;
	header.num_channels = nChannels;
	header.channels		= (EXRChannelInfo *) malloc(sizeof(EXRChannelInfo) * header.num_channels);
	// Must be (A)BGR order, since most of EXR viewers expect this channel order.
	strncpy(header.channels[0].name, "R", 255);
	header.channels[0].name[strlen("R")] = '\0';
	if (nChannels > 1) {
		strncpy(header.channels[1].name, "G", 255);
		header.channels[1].name[strlen("G")] = '\0';
	}
	if (nChannels > 2) {
		strncpy(header.channels[2].name, "B", 255);
		header.channels[2].name[strlen("B")] = '\0';
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

int load_exr(float **data, int *width, int *height, const char *filename, bool flip) {
	const char *err = nullptr;

	EXRVersion exr_version;

	int ret = ParseEXRVersionFromFile(&exr_version, filename);
	if (ret != 0) {
		std::string error_message = std::string("Failed to parse EXR image version");
		logError(error_message);
		return ret;
	}

	if (exr_version.multipart) {
		logError("EXR file must be singlepart");
		return ret;
	}

	// 2. Read EXR header
	EXRHeader exr_header;
	InitEXRHeader(&exr_header);

	err = nullptr; // or `nullptr` in C++11 or later.
	ret				= ParseEXRHeaderFromFile(&exr_header, &exr_version, filename, &err);
	if (ret != 0) {
		std::string error_message = std::string("Failed to parse EXR image header: ") + err;
		FreeEXRErrorMessage(err); // free's buffer for an error message
		logError(error_message);
		return ret;
	}

	bool full_precision = exr_header.pixel_types[0] == TINYEXR_PIXELTYPE_FLOAT;
	// Read FLOAT channel as HALF.
	for (int i = 0; i < exr_header.num_channels; i++) {
		bool local_fp = exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_FLOAT;
		if (local_fp != full_precision) {
			throw std::runtime_error("Can't handle EXR images with mixed channel types");
		}
	}

	if (flip)
		exr_header.line_order = !static_cast<bool>(exr_header.line_order);

	EXRImage exr_image;
	InitEXRImage(&exr_image);

	ret = LoadEXRImageFromFile(&exr_image, &exr_header, filename, &err);
	if (ret != 0) {
		std::string error_message = std::string("Failed to load EXR image: ") + err;
		FreeEXRHeader(&exr_header);
		FreeEXRErrorMessage(err); // free's buffer for an error message
		throw std::runtime_error(error_message);
	}

	// 3. Access image data
	// `exr_image.images` will be filled when EXR is scanline format.
	// `exr_image.tiled` will be filled when EXR is tiled format.

	*width	= exr_image.width;
	*height = exr_image.height;

	size_t n_pixels = exr_image.width * exr_image.height;

	size_t bytes_per_pixel = full_precision ? 4 : 2;

	std::vector<uchar> tmp(n_pixels * 4 * bytes_per_pixel, 0);

	bool has_alpha = false;
	for (int c = 0; c < exr_header.num_channels; c++) {
		if (strcmp(exr_header.channels[c].name, "R") == 0) {
			memcpy(tmp.data() + n_pixels * 0 * bytes_per_pixel, exr_image.images[c],
				   bytes_per_pixel * n_pixels);
		} else if (strcmp(exr_header.channels[c].name, "G") == 0) {
			memcpy(tmp.data() + n_pixels * 1 * bytes_per_pixel, exr_image.images[c],
				   bytes_per_pixel * n_pixels);
		} else if (strcmp(exr_header.channels[c].name, "B") == 0) {
			memcpy(tmp.data() + n_pixels * 2 * bytes_per_pixel, exr_image.images[c],
				   bytes_per_pixel * n_pixels);
		} else if (strcmp(exr_header.channels[c].name, "A") == 0) {
			has_alpha = true;
			memcpy(tmp.data() + n_pixels * 3 * bytes_per_pixel, exr_image.images[c],
				   bytes_per_pixel * n_pixels);
		}
	}
	
	*data = (float *) malloc(n_pixels * sizeof(Color4f));
	for (int pix = 0; pix < n_pixels; pix++) {
		float *rgba = *data + 4 * pix;
		rgba[0] = full_precision ? ((float *) tmp.data())[0 * n_pixels + pix]
								 : ((bfloat16 *) tmp.data())[0 * n_pixels + pix];
		rgba[1] = full_precision ? ((float *) tmp.data())[1 * n_pixels + pix]
							: ((bfloat16 *) tmp.data())[1 * n_pixels + pix];
		rgba[2] = full_precision ? ((float *) tmp.data())[2 * n_pixels + pix]
								 : ((bfloat16 *) tmp.data())[2 * n_pixels + pix];
		if (has_alpha)
			rgba[3] = full_precision ? ((float *) tmp.data())[3 * n_pixels + pix]
									 : ((bfloat16 *) tmp.data())[3 * n_pixels + pix];
		else
			rgba[3] = 1;
	}
	return 0;
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

KRR_NAMESPACE_END