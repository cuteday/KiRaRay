#include "image.h"
#include "logger.h"
#define TINYEXR_USE_MINIZ (0)
#define TINYEXR_IMPLEMENTATION
#include "zlib.h"
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

int load_exr(float **data, int *width, int *height, const char *filename, bool flip) {
	const char **err = nullptr;
	const char *layername = nullptr;

	EXRVersion exr_version;
	EXRImage exr_image;
	EXRHeader exr_header;
	InitEXRHeader(&exr_header);
	InitEXRImage(&exr_image);

	{
		int ret = ParseEXRVersionFromFile(&exr_version, filename);
		if (ret != TINYEXR_SUCCESS) {
			std::stringstream ss;
			ss << "Failed to open EXR file or read version info from EXR file. code(" << ret << ")";
			logError(ss.str());
			return ret;
		}

		if (exr_version.multipart || exr_version.non_image) {
			logError(
				"Loading multipart or DeepImage is not supported  in LoadEXR() API");
			return TINYEXR_ERROR_INVALID_DATA;
		}
	}

	{
		int ret = ParseEXRHeaderFromFile(&exr_header, &exr_version, filename, err);
		if (ret != TINYEXR_SUCCESS) {
			FreeEXRHeader(&exr_header);
			return ret;
		}
	}

	for (int i = 0; i < exr_header.num_channels; i++) {
		if (exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
			exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
		}
	}

	if (flip) exr_header.line_order = 1;

	{
		int ret = LoadEXRImageFromFile(&exr_image, &exr_header, filename, err);
		if (ret != TINYEXR_SUCCESS) {
			FreeEXRHeader(&exr_header);
			return ret;
		}
	}

	// RGBA
	int idxR = -1;
	int idxG = -1;
	int idxB = -1;
	int idxA = -1;

	std::vector<std::string> layer_names;
	::tinyexr::GetLayers(exr_header, layer_names);

	std::vector<::tinyexr::LayerChannel> channels;
	::tinyexr::ChannelsInLayer(exr_header, layername == NULL ? "" : std::string(layername), channels);

	if (channels.size() < 1) {
		logError("Layer Not Found");
		FreeEXRHeader(&exr_header);
		FreeEXRImage(&exr_image);
		return TINYEXR_ERROR_LAYER_NOT_FOUND;
	}

	size_t ch_count = channels.size() < 4 ? channels.size() : 4;
	for (size_t c = 0; c < ch_count; c++) {
		const ::tinyexr::LayerChannel &ch = channels[c];

		if (ch.name == "R") {
			idxR = int(ch.index);
		} else if (ch.name == "G") {
			idxG = int(ch.index);
		} else if (ch.name == "B") {
			idxB = int(ch.index);
		} else if (ch.name == "A") {
			idxA = int(ch.index);
		}
	}

	if (channels.size() == 1) {
		int chIdx = int(channels.front().index);
		// Grayscale channel only.

		(*data) = reinterpret_cast<float *>(
			malloc(4 * sizeof(float) * static_cast<size_t>(exr_image.width) *
				   static_cast<size_t>(exr_image.height)));

		if (exr_header.tiled) {
			for (int it = 0; it < exr_image.num_tiles; it++) {
				for (int j = 0; j < exr_header.tile_size_y; j++) {
					for (int i = 0; i < exr_header.tile_size_x; i++) {
						const int ii = exr_image.tiles[it].offset_x *
										   static_cast<int>(exr_header.tile_size_x) +
									   i;
						const int jj = exr_image.tiles[it].offset_y *
										   static_cast<int>(exr_header.tile_size_y) +
									   j;
						const int idx = ii + jj * static_cast<int>(exr_image.width);

						// out of region check.
						if (ii >= exr_image.width) {
							continue;
						}
						if (jj >= exr_image.height) {
							continue;
						}
						const int srcIdx		 = i + j * exr_header.tile_size_x;
						unsigned char **src		 = exr_image.tiles[it].images;
						(*data)[4 * idx + 0] = reinterpret_cast<float **>(src)[chIdx][srcIdx];
						(*data)[4 * idx + 1] = reinterpret_cast<float **>(src)[chIdx][srcIdx];
						(*data)[4 * idx + 2] = reinterpret_cast<float **>(src)[chIdx][srcIdx];
						(*data)[4 * idx + 3] = reinterpret_cast<float **>(src)[chIdx][srcIdx];
					}
				}
			}
		} else {
			for (int i = 0; i < exr_image.width * exr_image.height; i++) {
				const float val		   = reinterpret_cast<float **>(exr_image.images)[chIdx][i];
				(*data)[4 * i + 0] = val;
				(*data)[4 * i + 1] = val;
				(*data)[4 * i + 2] = val;
				(*data)[4 * i + 3] = val;
			}
		}
	} else {
		// Assume RGB(A)
		(*data) = reinterpret_cast<float *>(
			malloc(4 * sizeof(float) * static_cast<size_t>(exr_image.width) *
				   static_cast<size_t>(exr_image.height)));
		if (exr_header.tiled) {
			for (int it = 0; it < exr_image.num_tiles; it++) {
				for (int j = 0; j < exr_header.tile_size_y; j++) {
					for (int i = 0; i < exr_header.tile_size_x; i++) {
						const int ii  = exr_image.tiles[it].offset_x * exr_header.tile_size_x + i;
						const int jj  = exr_image.tiles[it].offset_y * exr_header.tile_size_y + j;
						const int idx = ii + jj * exr_image.width;

						// out of region check.
						if (ii >= exr_image.width) {
							continue;
						}
						if (jj >= exr_image.height) {
							continue;
						}
						const int srcIdx		 = i + j * exr_header.tile_size_x;
						unsigned char **src		 = exr_image.tiles[it].images;
						(*data)[4 * idx + 0] = reinterpret_cast<float **>(src)[idxR][srcIdx];
						(*data)[4 * idx + 1] = reinterpret_cast<float **>(src)[idxG][srcIdx];
						(*data)[4 * idx + 2] = reinterpret_cast<float **>(src)[idxB][srcIdx];
						if (idxA != -1) {
							(*data)[4 * idx + 3] =
								reinterpret_cast<float **>(src)[idxA][srcIdx];
						} else {
							(*data)[4 * idx + 3] = 1.0;
						}
					}
				}
			}
		} else {
			for (int i = 0; i < exr_image.width * exr_image.height; i++) {
				(*data)[4 * i + 0] = reinterpret_cast<float **>(exr_image.images)[idxR][i];
				(*data)[4 * i + 1] = reinterpret_cast<float **>(exr_image.images)[idxG][i];
				(*data)[4 * i + 2] = reinterpret_cast<float **>(exr_image.images)[idxB][i];
				if (idxA != -1) {
					(*data)[4 * i + 3] = reinterpret_cast<float **>(exr_image.images)[idxA][i];
				} else {
					(*data)[4 * i + 3] = 1.0;
				}
			}
		}
	}

	(*width)  = exr_image.width;
	(*height) = exr_image.height;

	FreeEXRHeader(&exr_header);
	FreeEXRImage(&exr_image);
	
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