#ifndef IMGTRANSPIECEWISEAFFINE_H
#define IMGTRANSPIECEWISEAFFINE_H

#include <vector>
#include <array>
#include "opencv2/imgproc.hpp"
#include "imgwarp_mls.h"

template < typename Type >
class ImgWarp_PieceWiseAffine : public ImgWarp_MLS<Type>
{

public:
    //! How to deal with the background.
    /*!
        BGNone: No background is reserved.
        BGMLS: Use MLS to deal with the background.
        BGPieceWise: Use the same scheme for the background.
    */
    enum BGFill {
			BGNone, //! No background is reserved.
            BGMLS,  //! Use MLS to deal with the background.
			BGPieceWise}; //! Use the same scheme for the background.
    

	ImgWarp_PieceWiseAffine(void);

	cv::Point_<Type> getMLSDelta(int x, int y);

	void calcDelta();

    BGFill backGroundFillAlg;
private:
 

	//! Find the Delaunay division for given points(Return in int coordinates).
	std::vector<cv::Vec6f> delaunayDiv(const std::vector<cv::Point_<Type>> &vP, cv::Rect boundRect);

	//trait function 
	void subdivinsert(cv::Subdiv2D & subdiv, const std::vector<cv::Point_<Type>> &vP);

};

//-------------------------------------------------------------------
//Type define 
using ImgWarpPieceWiseAffine = ImgWarp_PieceWiseAffine<float>;

//-------------------------------------------------------------------
//Template Class implementation

template <typename Type>
ImgWarp_PieceWiseAffine<Type>::ImgWarp_PieceWiseAffine(void) 
{
	backGroundFillAlg = BGNone;
	VirtualCalcDelta = [this]{ this->calcDelta(); };
}

template <typename Type>
void ImgWarp_PieceWiseAffine<Type>::subdivinsert(cv::Subdiv2D & subdiv, const std::vector<cv::Point_<Type>> &vP){
	std::vector<cv::Point2f> vPoints;
	vPoints.assign(vP.begin(), vP.end());
	subdiv.insert(vPoints);
};
template <> //Float optimisation
void ImgWarp_PieceWiseAffine<float>::subdivinsert(cv::Subdiv2D & subdiv, const std::vector<cv::Point_<float>> &vP)
{
	subdiv.insert(vP);
}

template <typename Type>
std::vector<cv::Vec6f> ImgWarp_PieceWiseAffine<Type>::delaunayDiv(const std::vector<cv::Point_<Type>> &vP, cv::Rect boundRect)
{
	cv::Subdiv2D subdiv(boundRect);
	subdivinsert(subdiv, vP);
	std::vector<cv::Vec6f> ans;
	subdiv.getTriangleList(ans);

	//Triangle filter, some triangles are outside the init rectangle
	//--------------------------------------
	std::vector<cv::Vec6f> vTriangles;
	vTriangles.reserve(ans.size());
	auto Vec6fToVVec2f = [](const cv::Vec6f & v6)
	{
		return std::vector <cv::Point2f>
		{
			{ v6[0], v6[1] },
			{ v6[2], v6[3] },
			{ v6[4], v6[5] },
		};
	};

	for (auto & t : ans)
	{
		const auto & p = Vec6fToVVec2f(t);
		cv::Rect trect = cv::boundingRect(p);
		if ((boundRect & trect) == trect)
			vTriangles.emplace_back(t);
	}

	return vTriangles;
}

template <typename Type>
cv::Point_<Type> ImgWarp_PieceWiseAffine<Type>::getMLSDelta(int x, int y)
{
	static cv::Point_<Type> swq, qstar, newP, tmpP;
	Type sw;

	static std::vector<Type> w;
	w.resize(nPoint);

	static cv::Point_<Type> swp, pstar, curV, curVJ, Pi, PiJ;
	Type miu_s;

	int i = x;
	int j = y;
	int k;

	sw = 0;
	swp.x = swp.y = 0;
	swq.x = swq.y = 0;
	newP.x = newP.y = 0;
	curV.x = i;
	curV.y = j;
	for (k = 0; k < nPoint; k++) {
		if ((i == oldDotL[k].x) && j == oldDotL[k].y) break;
		w[k] = 1 / ((i - oldDotL[k].x) * (i - oldDotL[k].x) +
			(j - oldDotL[k].y) * (j - oldDotL[k].y));
		sw = sw + w[k];
		swp = swp + w[k] * oldDotL[k];
		swq = swq + w[k] * newDotL[k];
	}
	if (k == nPoint) {
		pstar = (1 / sw) * swp;
		qstar = 1 / sw * swq;

		// Calc miu_s
		miu_s = 0;
		for (k = 0; k < nPoint; k++) {
			if (i == oldDotL[k].x && j == oldDotL[k].y) continue;

			Pi = oldDotL[k] - pstar;
			miu_s += w[k] * Pi.dot(Pi);
		}

		curV -= pstar;
		curVJ.x = -curV.y, curVJ.y = curV.x;

		for (k = 0; k < nPoint; k++) {
			if (i == oldDotL[k].x && j == oldDotL[k].y) continue;

			Pi = oldDotL[k] - pstar;
			PiJ.x = -Pi.y, PiJ.y = Pi.x;

			tmpP.x = Pi.dot(curV) * newDotL[k].x - PiJ.dot(curV) * newDotL[k].y;
			tmpP.y =
				-Pi.dot(curVJ) * newDotL[k].x + PiJ.dot(curVJ) * newDotL[k].y;
			tmpP *= w[k] / miu_s;
			newP += tmpP;
		}
		newP += qstar;
	}
	else {
		newP = newDotL[k];
	}

	newP.x -= i;
	newP.y -= j;
	return newP;
}

template <typename Type>
void ImgWarp_PieceWiseAffine<Type>::calcDelta() 
{
	cv::Mat_<int> imgLabel = cv::Mat_<int>::zeros(tarH, tarW);

	rDx = rDx.zeros(tarH, tarW);
	rDy = rDy.zeros(tarH, tarW);
	for (int i = 0; i < this->nPoint; i++) {
		//! Ignore points outside the target image
		if (oldDotL[i].x < 0) oldDotL[i].x = 0;
		if (oldDotL[i].y < 0) oldDotL[i].y = 0;
		if (oldDotL[i].x >= tarW) oldDotL[i].x = tarW - 1;
		if (oldDotL[i].y >= tarH) oldDotL[i].y = tarH - 1;

		rDx(oldDotL[i]) = newDotL[i].x - oldDotL[i].x;
		rDy(oldDotL[i]) = newDotL[i].y - oldDotL[i].y;
	}
	rDx(0, 0) = rDy(0, 0) = 0;
	rDx(tarH - 1, 0) = rDy(0, tarW - 1) = 0;
	rDy(tarH - 1, 0) = rDy(tarH - 1, tarW - 1) = srcH - tarH;
	rDx(0, tarW - 1) = rDx(tarH - 1, tarW - 1) = srcW - tarW;

	std::vector<cv::Vec6f> V;
	std::vector<cv::Vec6f>::iterator it;
	cv::Rect_<int> boundRect(0, 0, tarW, tarH);
	std::vector<cv::Point_<Type> > oL1 = oldDotL;
	if (backGroundFillAlg == BGPieceWise) {
		oL1.push_back(cv::Point2d(0, 0));
		oL1.push_back(cv::Point2d(0, tarH - 1));
		oL1.push_back(cv::Point2d(tarW - 1, 0));
		oL1.push_back(cv::Point2d(tarW - 1, tarH - 1));
	}
	// In order preserv the background
	V = delaunayDiv(oL1, boundRect);

	auto roundp2f = [](const cv::Vec6f & p)->std::array <cv::Point, 3> {
		std::array <cv::Point, 3> vp = {
			cv::Point((int)std::round(p[0]), (int)std::round(p[1])),
			cv::Point((int)std::round(p[2]), (int)std::round(p[3])),
			cv::Point((int)std::round(p[4]), (int)std::round(p[5]))
		};
		return vp;
	};


	cv::Mat_<uchar> imgTmp = cv::Mat_<uchar>::zeros(tarH, tarW);
	for (it = V.begin(); it != V.end(); it++) {

		auto v = roundp2f(*it);

		cv::line(imgTmp, v[0], v[1], 255, 1, CV_AA);
		cv::line(imgTmp, v[0], v[2], 255, 1, CV_AA);
		cv::line(imgTmp, v[2], v[1], 255, 1, CV_AA);

		// Not interested in points outside the region.
		if (!(v[0].inside(boundRect) && v[1].inside(boundRect) &&
			v[2].inside(boundRect)))
			continue;

		cv::fillConvexPoly(imgLabel, &v[0], 3,
			cv::Scalar_<int>(it - V.begin() + 1));
	}
	// imshow("imgTmp", imgTmp);
	// cvWaitKey(10);

	int i, j;

	cv::Point_<float> v1, v2, curV;

	for (i = 0;; i += gridSize) {
		if (i >= tarW && i < tarW + gridSize - 1)
			i = tarW - 1;
		else if (i >= tarW)
			break;
		for (j = 0;; j += gridSize) {
			if (j >= tarH && j < tarH + gridSize - 1)
				j = tarH - 1;
			else if (j >= tarH)
				break;
			int tId = imgLabel(j, i) - 1;
			if (tId < 0) {
				if (backGroundFillAlg == BGMLS) {
					cv::Point_<Type> dV = getMLSDelta(i, j);
					rDx(j, i) = dV.x;
					rDy(j, i) = dV.y;
				}
				else {
					rDx(j, i) = -i;
					rDy(j, i) = -j;
				}
				continue;
			}

			cv::Point_<float> * vtid = (cv::Point_<float>*)(&V[tId][0]);
			v1 = vtid[1] - vtid[0];
			v2 = vtid[2] - vtid[0];
			curV.x = i, curV.y = j;
			curV -= vtid[0];

			Type d0, d1, d2;
			d2 = Type(v1.x * curV.y - curV.x * v1.y) /
				(v1.x * v2.y - v2.x * v1.y);
			d1 = Type(v2.x * curV.y - curV.x * v2.y) /
				(v2.x * v1.y - v1.x * v2.y);
			d0 = 1 - d1 - d2;
			rDx(j, i) = d0 * rDx(vtid[0]) + d1 * rDx(vtid[1]) +
				d2 * rDx(vtid[2]);
			rDy(j, i) = d0 * rDy(vtid[0]) + d1 * rDy(vtid[1]) +
				d2 * rDy(vtid[2]);
		}
	}
}





#endif //IMGTRANSPIECEWISEAFFINE_H