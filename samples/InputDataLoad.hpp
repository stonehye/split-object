
#ifndef InputDataLoad_hpp
#define InputDataLoad_hpp

#include <filesystem>
#include <fstream>
#include <sstream>
#include <optional>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace Merge
{
	struct InputData
	{
		InputData() : depth_raw(NULL), preprocessed(false) {}

		~InputData()
		{
			if (depth_raw)
			{
				delete depth_raw;
				depth_raw = NULL;
			}
		}

		struct Vector3
		{
			float x{ 0.0f };
			float y{ 0.0f };
			float z{ 0.0f };
			Vector3() = default;
			Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
			Vector3(std::vector<std::string> accel_str_list) : x(stof(accel_str_list[0])), y(stof(accel_str_list[1])), z(stof(accel_str_list[2])) {}
		} accel;

		cv::Mat rgb;
		cv::Mat depth;
		float* depth_raw;
		cv::Mat confidence;
		cv::Mat sailency;
		cv::Mat object_mask;

		std::string file_name;

		bool preprocessed;

		void Preprocess_input()
		{
			if (preprocessed == false)
			{
				cv::flip(rgb.t(), rgb, -1);
				cv::flip(confidence.t(), confidence, -1);
				cv::flip(depth.t(), depth, -1);
				depth.convertTo(depth, CV_32FC1);
				depth.convertTo(depth, CV_32FC1, 0.001f);

				preprocessed = true;
			}
		}

		void Visualize()
		{
			cv::Mat rgb_vis, depth_vis, confidence_vis;

			confidence_vis = confidence.clone();

			if (preprocessed)
			{
				cv::flip(rgb.clone().t(), rgb_vis, -1);
				cv::flip(depth.clone().t(), depth_vis, -1);
				cv::flip(confidence_vis.t(), confidence_vis, -1);
			}
			else
			{
				rgb_vis = rgb.clone();
				depth.convertTo(depth_vis, CV_32FC1);
				depth_vis.convertTo(depth_vis, CV_32FC1, 0.001f);
			}

			cv::imshow("rgb", rgb_vis);
			cv::imshow("depth", depth_vis);
			cv::imshow("confidence", confidence_vis*128);
			cv::imshow("mask", object_mask);
			cv::waitKey(0);

			cv::imwrite("rgb.jpg", rgb_vis);
			cv::imwrite("confidence.jpg", confidence_vis * 128);
		}
	};
	
	typedef std::pair<InputData, int> InputDataLabel;

	class InputDataLoader
	{
	public:
		static std::vector<InputDataLabel> Load_all_with_label(const std::string& root_dir, const char delimeter = '%')
		{
			std::vector<InputDataLabel> data_list;

			for (const auto& entry : fs::directory_iterator(root_dir))
			{
				const auto& folder_label = entry.path().generic_string();
				const int label = stoi(entry.path().filename().string());

				const auto& samples = Load_all(folder_label);
				
				for (auto& sample : samples)
					data_list.push_back({ *sample.get(), label });
			}

			return data_list;
		}

		static std::vector<std::shared_ptr<InputData>> Load_all(const std::string& root_dir, const char delimeter = '%')
		{
			std::vector<std::shared_ptr<InputData>> data_list;
			std::set<std::string> file_path_set;

			for (const auto& entry : fs::directory_iterator(root_dir))
			{
				const auto& file_path_str = entry.path().generic_string();
				std::istringstream ss(file_path_str);
				std::string split_file_path;

				while (getline(ss, split_file_path, delimeter))
				{
					file_path_set.insert(split_file_path);
					break;
				}
			}
			for (const auto& file_path : file_path_set)
			{
				std::shared_ptr<InputData> data = std::make_shared<InputData>();
				if(Load_single(file_path, data))
					data_list.push_back(data);
			}

			return data_list;
		}

		static bool Load_single(const std::string& file_path, std::shared_ptr<InputData>& data)
		{
			const auto& accel = _Load_acceleration(file_path + acceleration_ext);

			if (!accel.has_value())
				return false;

			if (!_Load_images(file_path, data->rgb, data->depth, data->confidence, data->sailency, data->object_mask))
				return false;

			data->accel = accel.value();
			data->file_name = file_path;
			data->depth_raw = _Load_raw_depth(file_path + depth_raw_ext);

			return true;
		}

	private:
		static float* _Load_raw_depth(const std::string& filename)
		{
			std::ifstream file(filename, std::ios::in | std::ios::binary);
			float* depth_raw = NULL;
			
			if (file) {
				file.seekg(0, file.end);
				int length = (int)file.tellg() / 4;
				file.seekg(0, file.beg);
				std::cout << length << std::endl;
				depth_raw = new float[length];

				file.read((char*)depth_raw, length * 4);
				std::cout << depth_raw[0] << std::endl;
				file.close();
			}
			
			return depth_raw;
		}

		static std::optional<InputData::Vector3> _Load_acceleration(const std::string& file_path)
		{
			std::ifstream accel_file;
			accel_file.open(file_path);

			char arr[64];
			std::vector<std::string> accel_list;
			if (!accel_file.is_open())
				return std::nullopt;

			while (accel_file.getline(arr, 64, ','))
			{
				accel_list.push_back(std::string(arr));
			}

			return std::optional<InputData::Vector3>{ InputData::Vector3(accel_list) };
		}

		static bool _Load_images(const std::string& file_path, cv::Mat& rgb, cv::Mat& depth, cv::Mat& confidence)
		{
			rgb = cv::imread(file_path + rgb_ext);
			depth = cv::imread(file_path + depth_ext, cv::IMREAD_ANYDEPTH);
			confidence = cv::imread(file_path + confidence_ext, cv::IMREAD_GRAYSCALE);

			if (rgb.empty() || depth.empty() || confidence.empty())
				return false;

			return true;
		}

		static bool _Load_images(const std::string& file_path, cv::Mat& rgb, cv::Mat& depth, cv::Mat& confidence, cv::Mat& sailency)
		{
			if (!_Load_images(file_path, rgb, depth, confidence))
				return false;
			
			sailency = cv::imread(file_path + sailency_ext, cv::IMREAD_GRAYSCALE);

			if (sailency.empty())
				return false;

			return true;
		}

		static bool _Load_images(const std::string& file_path, cv::Mat& rgb, cv::Mat& depth, cv::Mat& confidence, cv::Mat& sailency, cv::Mat& object_mask)
		{
			if(!_Load_images(file_path, rgb, depth, confidence, sailency))
				return false;

			object_mask = cv::imread(file_path + object_mask_ext, cv::IMREAD_GRAYSCALE);

			if (object_mask.empty())
				return false;

			return true;
		}

	private:
		static std::string rgb_ext;
		static std::string depth_ext;
		static std::string depth_raw_ext;
		static std::string confidence_ext;
		static std::string sailency_ext;
		static std::string object_mask_ext;
		static std::string acceleration_ext;
	};
	
}

void test_load_input_data();

#endif