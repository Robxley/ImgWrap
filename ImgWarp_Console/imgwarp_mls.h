#ifndef IMGTRANS_MLS_H
#define IMGTRANS_MLS_H
#include <vector>
#include <functional>
#include "opencv2/opencv.hpp"


#include "Threads.h"

//! The base class for Moving Least Square image warping.
/*!
 * Choose one of the subclasses, the easiest interface to generate
 * an output is to use setAllAndGenerate function.

 * Type: Accuracy of calculations
 */


template <typename Type>
class ImgWarp_MLS {
   public:
	   ImgWarp_MLS(){};
   
    //! Set all and generate an output.
    /*!
      \param oriImg the image to be warped.
      \param qsrc A list of "from" points.
      \param qdst A list of "target" points.
      \param outW The width of the output image.
      \param outH The height of the output image.
      \param transRatio 1 means warp to target points, 0 means no warping

      This will do all the initialization and generate a warped image.
      After calling this, one can later call genNewImg with different
      transRatios to generate a warping animation.
    */


	   cv::Mat setAllAndGenerate(const cv::Mat &oriImg,
		   const std::vector<cv::Point_<Type> > &qsrc,
		   const std::vector<cv::Point_<Type> > &qdst,
		   const int outW, const int outH,
		   const double transRatio = 1, int iNThreads = 8);

	   cv::Mat setAllAndGenerate(const cv::Mat &oriImg, cv::Mat &newImg,
		   const std::vector<cv::Point_<Type> > &qsrc,
		   const std::vector<cv::Point_<Type> > &qdst,
		   const double transRatio = 1, int iNThreads = 8);

	//! Set the list of target points
	void setSrcPoints(const std::vector<cv::Point_<Type> > &qsrc);

	//! Set the list of source points
	void setDstPoints(const std::vector<cv::Point_<Type> > &qdst);

	void SetAlphaMLS(Type a) {	alpha = a;	}
	void SetGridSize(int s) { gridSize = s; }

protected:

    //! Generate the warped image.
    /*! This function generate a warped image using PRE-CALCULATED data.
     *  DO NOT CALL THIS AT FIRST! Call this after at least one call of
     *  setAllAndGenerate.
	 *  T : image depth,
	 *  Type : Accuracy of calculations
     */

	template <typename T>
	cv::Mat genNewImg(const cv::Mat &oriImg, cv::Mat & newImg, double transRatio = 1);

    //! Calculate delta value which will be used for generating the warped
    //image.
	//virtual void calcDelta();
	std::function<void()> VirtualCalcDelta;

    //! Parameter for MLS.
	Type alpha = 1;

    //! Parameter for MLS.
    int gridSize = 5;

    //! The size of the original image. For precalculation.
    void setSize(int w, int h) { srcW = w, srcH = h; }

    //! The size of output image
    void setTargetSize(const int outW, const int outH) {
        tarW = outW;
        tarH = outH;
    }

   protected:

	Type bilinear_interp(Type x, Type y, Type v11, Type v12, Type v21, Type v22) {
		   return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x;
	   }

    std::vector<cv::Point_<Type> > oldDotL, newDotL;

    int nPoint;

	cv::Mat_<Type> /*! \brief delta_x */ rDx, /*! \brief delta_y */ rDy;

    int srcW, srcH;
    int tarW, tarH;

	int m_iNThreads = 8;
};

//-------------------------------------------------------------------
//Type define 
using ImgWarpMLS = ImgWarp_MLS<float>;


//-------------------------------------------------------------------
//Class implementation

template <typename Type>
cv::Mat ImgWarp_MLS<Type>::setAllAndGenerate(const cv::Mat &oriImg,
	const std::vector<cv::Point_<Type> > &qsrc,
	const std::vector<cv::Point_<Type> > &qdst,
	const int outW, const int outH,
	const double transRatio, int iNThreads)
{
	

	m_iNThreads = iNThreads;
	setSize(oriImg.cols, oriImg.rows);
	setTargetSize(outW, outH);
	setSrcPoints(qsrc);
	setDstPoints(qdst);
	VirtualCalcDelta();

	cv::Mat result;
	switch (oriImg.depth())
	{
	case CV_8S:		result = genNewImg<char>(oriImg, cv::Mat(), transRatio); break;
	case CV_8U:		result = genNewImg<uchar>(oriImg, cv::Mat(), transRatio); break;
	case CV_16S:	result = genNewImg<short>(oriImg, cv::Mat(), transRatio); break;
	case CV_16U:	result = genNewImg<ushort>(oriImg, cv::Mat(), transRatio); break;
	case CV_32F:	result = genNewImg<float>(oriImg, cv::Mat(), transRatio); break;
	case CV_32S:	result = genNewImg<int>(oriImg, cv::Mat(), transRatio); break;
	case CV_64F:	result = genNewImg<double>(oriImg, cv::Mat(), transRatio); break;
	default:

		break;
	}
	return result;
	

}

template <typename Type>
cv::Mat ImgWarp_MLS<Type>::setAllAndGenerate(const cv::Mat &oriImg, cv::Mat &newImg,
	const std::vector<cv::Point_<Type> > &qsrc,
	const std::vector<cv::Point_<Type> > &qdst,
	const double transRatio, int iNThreads)
{
	m_iNThreads = iNThreads;
	setSize(oriImg.cols, oriImg.rows);
	if (newImg.empty())
		setTargetSize(oriImg.cols, oriImg.rows);
	else
		setTargetSize(newImg.cols, newImg.rows);

	setSrcPoints(qsrc);
	setDstPoints(qdst);
	VirtualCalcDelta();

	switch (oriImg.depth())
	{
	case CV_8S:		genNewImg<char>(oriImg, newImg, transRatio); break;
	case CV_8U:		genNewImg<uchar>(oriImg, newImg, transRatio); break;
	case CV_16S:	genNewImg<short>(oriImg, newImg, transRatio); break;
	case CV_16U:	genNewImg<ushort>(oriImg, newImg, transRatio); break;
	case CV_32F:	genNewImg<float>(oriImg, newImg, transRatio); break;
	case CV_32S:	genNewImg<int>(oriImg, newImg, transRatio); break;
	case CV_64F:	genNewImg<double>(oriImg, newImg, transRatio); break;
	default:
		break;
	}

	return newImg;
}


template <typename Type>
void ImgWarp_MLS<Type>::setSrcPoints(const std::vector<cv::Point_<Type> > &qsrc)
{
	nPoint = qsrc.size();
	newDotL.clear();
	newDotL.assign(qsrc.begin(), qsrc.end());
}

template <typename Type>
void ImgWarp_MLS<Type>::setDstPoints(const std::vector<cv::Point_<Type> > &qdst)
{
	nPoint = qdst.size();
	oldDotL.clear();
	oldDotL.assign(qdst.begin(), qdst.end());
}

template <typename Type>
template <typename T>
cv::Mat ImgWarp_MLS<Type>::genNewImg(const cv::Mat &oriImg, cv::Mat & newImg, double transRatio)
{
	
	if (newImg.empty())
		newImg = cv::Mat(tarH, tarW, oriImg.type());
	else if(newImg.size() != cv::Size(tarW, tarH))
	{
		std::cout << "Wrong size"<<std::endl;
	}

	//Modif pour le multithread
	const int iChannel = oriImg.channels();
	const int kEnd = std::ceil((float)tarH / gridSize);

	MultiThreadFor(0,kEnd,m_iNThreads,[&](int k,int)
	//for (int i = 0; i < tarH; i += gridSize)  <- code d'origine
	{
		int i = k*gridSize;

		for (int j = 0; j < tarW; j += gridSize)
		{
			int ni = i + gridSize;
			int nj = j + gridSize;
			Type w = gridSize;
			Type h = gridSize;
			if (ni >= tarH) ni = tarH - 1, h = ni - i + 1;
			if (nj >= tarW) nj = tarW - 1, w = nj - j + 1;
			for (Type di = 0; di < h; di++)
				for (Type dj = 0; dj < w; dj++)
				{
					Type dih = di / h;
					Type djw = dj / w;
					Type deltaX =
						bilinear_interp(dih, djw, rDx(i, j), rDx(i, nj), rDx(ni, j), rDx(ni, nj));
					Type deltaY =
						bilinear_interp(dih, djw, rDy(i, j), rDy(i, nj), rDy(ni, j), rDy(ni, nj));
					Type nx = j + dj + deltaX * transRatio;
					Type ny = i + di + deltaY * transRatio;
					if (nx > srcW - 1) nx = srcW - 1;
					if (ny > srcH - 1) ny = srcH - 1;
					if (nx < 0) nx = 0;
					if (ny < 0) ny = 0;
					int nxi = int(nx);
					int nyi = int(ny);
					int nxi1 = ceil(nx);
					int nyi1 = ceil(ny);

					if (iChannel == 1)
						newImg.at<T>(i + di, j + dj) = 
						bilinear_interp(
						ny - nyi, nx - nxi, 
						oriImg.at<T>(nyi, nxi),
						oriImg.at<T>(nyi, nxi1),
						oriImg.at<T>(nyi1, nxi),
						oriImg.at<T>(nyi1, nxi1));
					else {
						for (int ll = 0; ll < 3; ll++)
							newImg.at<cv::Vec<T, 3>>(i + di, j + dj)[ll] =
							bilinear_interp(
							ny - nyi, nx - nxi,
							oriImg.at<cv::Vec<T, 3>>(nyi, nxi)[ll],
							oriImg.at<cv::Vec<T, 3>>(nyi, nxi1)[ll],
							oriImg.at<cv::Vec<T, 3>>(nyi1, nxi)[ll],
							oriImg.at<cv::Vec<T, 3>>(nyi1, nxi1)[ll]);
					}
				}
		}
	});
	


	return newImg;
	
}



#endif  // IMGTRANS_MLS_H
