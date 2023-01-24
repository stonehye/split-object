
/*

This class implements a very fast ellipse detector, codename: YAED (Yet Another Ellipse Detector)

*/

#ifndef CNEllipseDetector_h
#define CNEllipseDetector_h

#include "CNEllipseDetectorUtils.h"

using namespace std;
//using namespace cv;


#ifndef GLOBAL
#define GLOBAL

	extern bool myselect1;
	extern bool myselect2;
	extern bool myselect3;
	extern float tCNC;
	extern float tTCNl;
#endif 

#define DISCARD_CONSTRAINT_CONVEXITY		// 1, 2
//#define DISCARD_CONSTRAINT_POSITION
//#define DISCARD_CONSTRAINT_CENTER			// 1, 2

struct EllipseData
{
	bool isValid;
	float ta;
	float tb;
	float ra;
	float rb;
    cv::Point2f Ma;
    cv::Point2f Mb;
    cv::Point2f Cab;
	vector<float> Sa;
	vector<float> Sb;
};


class CNEllipseDetector
{
	// Preprocessing - Gaussian filter. See Sect [] in the paper
    cv::Size	_szPreProcessingGaussKernelSize;	// size of the Gaussian filter in preprocessing step
	double	_dPreProcessingGaussSigma;			// sigma of the Gaussian filter in the preprocessing step
		
	
	// Selection strategy - Step 1 - Discard noisy or straight arcs. See Sect [] in the paper
	int		_iMinEdgeLength;					// minimum edge size				
	float	_fMinOrientedRectSide;				// minumum size of the oriented bounding box containing the arc
	float	_fMaxRectAxesRatio;					// maximum aspect ratio of the oriented bounding box containing the arc

	// Selection strategy - Step 2 - Remove according to mutual convexities. See Sect [] in the paper
	float _fThPosition;

	// Selection Strategy - Step 3 - Number of points considered for slope estimation when estimating the center. See Sect [] in the paper
	unsigned _uNs;									// Find at most Ns parallel chords.

	// Selection strategy - Step 3 - Discard pairs of arcs if their estimated center is not close enough. See Sect [] in the paper
	float	_fMaxCenterDistance;				// maximum distance in pixel between 2 center points
	float	_fMaxCenterDistance2;				// _fMaxCenterDistance * _fMaxCenterDistance

	// Validation - Points within a this threshold are considered to lie on the ellipse contour. See Sect [] in the paper
	float	_fDistanceToEllipseContour;			// maximum distance between a point and the contour. See equation [] in the paper

	// Validation - Assign a score. See Sect [] in the paper
	float	_fMinScore;							// minimum score to confirm a detection
	float	_fMinReliability;					// minimum auxiliary score to confirm a detection


	// auxiliary variables
    cv::Size	_szImg;			// input image size

	double _myExecTime;		// execution time

	int ACC_N_SIZE;			// size of accumulator N = B/A
	int ACC_R_SIZE;			// size of accumulator R = rho = atan(K)
	int ACC_A_SIZE;			// size of accumulator A

	int* accN;				// pointer to accumulator N
	int* accR;				// pointer to accumulator R
	int* accA;				// pointer to accumulator A

	
public:
	float countsOfFindEllipse;
	float countsOfGetFastCenter;
	//Constructor and Destructor
	CNEllipseDetector(void);
	~CNEllipseDetector(void);

	void DetectAfterPreProcessing(vector<EllipseM>& ellipses, cv::Mat1b& E, cv::Mat1f& PHI);

	//Detect the ellipses in the gray image
	void Detect(cv::Mat1b& gray, vector<EllipseM>& ellipses, const int rgb_height, const int rgb_center_x, const int rgb_center_y);
	
	//Draw the first iTopN ellipses on output
	void DrawDetectedEllipses(cv::Mat& output, vector<EllipseM>& ellipses, int iTopN=0, int thickness=2);
	
    void DrawDetectedEllipses(cv::Mat& output, EllipseM* ellipses, const int ellipses_size);
    
private:

	static const ushort PAIR_12 = 0x00;
	static const ushort PAIR_23 = 0x01;
	static const ushort PAIR_34 = 0x02;
	static const ushort PAIR_14 = 0x03;

	//generate keys from pair and indicse
	uint inline GenerateKey(uchar pair, ushort u, ushort v);

	void PrePeocessing(cv::Mat1b& I, cv::Mat1b& DP, cv::Mat1b& DN);

	void RemoveShortEdges(cv::Mat1b& edges, cv::Mat1b& clean);

	void ClusterEllipses(vector<EllipseM>& ellipses);

	int FindMaxK(const int* v) const;
	int FindMaxN(const int* v) const;
	int FindMaxA(const int* v) const;

	float GetMedianSlope(vector<cv::Point2f>& med, cv::Point2f& M, vector<float>& slopes);
	void GetFastCenter(vector<cv::Point>& e1, vector<cv::Point>& e2, EllipseData& data);

	void DetectEdges13(cv::Mat1b& DP, VVP& points_1, VVP& points_3);
	void DetectEdges24(cv::Mat1b& DN, VVP& points_2, VVP& points_4);

	void FindEllipses	(	cv::Point2f& center,
							VP& edge_i,
							VP& edge_j,
							VP& edge_k,
							EllipseData& data_ij,
							EllipseData& data_ik,
							vector<EllipseM>& ellipses
						);

    cv::Point2f GetCenterCoordinates(EllipseData& data_ij, EllipseData& data_ik);


	void Triplets124	(	VVP& pi,
							VVP& pj,
							VVP& pk,
							unordered_map<uint, EllipseData>& data,
							vector<EllipseM>& ellipses
						);

	void Triplets231	(	VVP& pi,
							VVP& pj,
							VVP& pk,
							unordered_map<uint, EllipseData>& data,
							vector<EllipseM>& ellipses
						);

	void Triplets342	(	VVP& pi,
							VVP& pj,
							VVP& pk,
							unordered_map<uint, EllipseData>& data,
							vector<EllipseM>& ellipses
						);

	void Triplets413	(	VVP& pi,
							VVP& pj,
							VVP& pk,
							unordered_map<uint, EllipseData>& data,
							vector<EllipseM>& ellipses
						);
};

#endif
