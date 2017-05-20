#include <iostream>
#include "opencv2/opencv.hpp"
#include "imgwarp_mls_rigid.h"
#include "imgwarp_mls_similarity.h"
#include "imgwarp_piecewiseaffine.h"

int main(int argc, char* argv[])
{
	if (argc <= 1)
		return 0;

	cv::Mat oInMat = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
	auto iSize = oInMat.size();

	float iN = 10;
	float wstep = (float)iSize.width / (iN);
	float hstep = (float)iSize.height / (iN);

	cv::RNG rng(12345);
	auto RNGPoint2f = [&]()
	{
		const float deform = 0.3f;
		return cv::Point2f(rng.uniform(-wstep, wstep), rng.uniform(-hstep, hstep))*deform;
	};



	std::vector<cv::Point2f> vInPoints, vOutPoints;
	vInPoints.reserve(iN*iN);
	vOutPoints.reserve(iN*iN);
	for (int i = 0; i <= iN; i++)
	{
		for (int j = 0; j <= iN; j++)
		{
			cv::Point2f p(i*wstep, j*hstep);
			vOutPoints.emplace_back(p + RNGPoint2f());
			vInPoints.emplace_back(p);
		}
	}

	auto outRect = cv::boundingRect(vOutPoints);
	//ImgWarpPieceWiseAffine imgTrans;
	ImgWarpSimilarity imgTrans;
	//ImgWarpRigid imgTrans;
	imgTrans.SetGridSize(5);
	imgTrans.SetAlphaMLS(1);

	cv::Mat oOutImg;
	imgTrans.setAllAndGenerate(oInMat, oOutImg, vInPoints, vOutPoints, 1,4);

	if (argc > 2)
		cv::imwrite(argv[1], oOutImg);
	else
	{
		cv::imshow("input", oInMat);
		cv::imshow("img warp", oOutImg);
	}

	cv::waitKey(0);

	system("Pause");
    return 0;
}

