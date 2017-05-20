/*
   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public
   License version 2 as published by the Free Software Foundation.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public License
   along with this library; see the file COPYING.LIB.  If not, write to
   the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.
*/

#ifndef IMGTRANS_MLS_SIMILARITY_H
#define IMGTRANS_MLS_SIMILARITY_H
#include "imgwarp_mls.h"
#include "opencv2/opencv.hpp"
#include <vector>

template<typename T> class ImgWarp_MLS_Similarity;


//! The class for MLS Similarity transform.
template <typename Type>
class ImgWarp_MLS_Similarity : public ImgWarp_MLS<Type>
{
   public:
	   ImgWarp_MLS_Similarity();
	   void calcDelta();
};

//-------------------------------------------------------------------
//Type define 
using ImgWarpSimilarity = ImgWarp_MLS_Similarity<float>;

//-------------------------------------------------------------------
//Template Class implementation

template <typename Type>
ImgWarp_MLS_Similarity<Type>::ImgWarp_MLS_Similarity()
{
	VirtualCalcDelta = [this]{this->calcDelta(); };
}


template <typename Type>
void ImgWarp_MLS_Similarity<Type>::calcDelta()
{
	int i, j, k;

	cv::Point_<Type> swq, qstar, newP, tmpP;
	Type sw;

	std::vector<Type> w(nPoint);

	rDx.create(tarH, tarW);
	rDy.create(tarH, tarW);

	if (nPoint < 2) {
		rDx.setTo(0);
		rDy.setTo(0);
		return;
	}

	cv::Point_<Type> swp, pstar, curV, curVJ, Pi, PiJ;
	Type miu_s;

	for (i = 0;; i += gridSize)
	{
		if (i >= tarW && i < tarW + gridSize - 1)
			i = tarW - 1;
		else if (i >= tarW)
			break;
		for (j = 0;; j += gridSize) {
			if (j >= tarH && j < tarH + gridSize - 1)
				j = tarH - 1;
			else if (j >= tarH)
				break;
			sw = 0;
			swp.x = swp.y = 0;
			swq.x = swq.y = 0;
			newP.x = newP.y = 0;
			curV.x = i;
			curV.y = j;
			for (k = 0; k < nPoint; k++) 
			{
				if ((i == oldDotL[k].x) && j == oldDotL[k].y) break;
				w[k] = 1 / ((i - oldDotL[k].x) * (i - oldDotL[k].x) +
					(j - oldDotL[k].y) * (j - oldDotL[k].y));
				sw = sw + w[k];
				swp = swp + w[k] * oldDotL[k];
				swq = swq + w[k] * newDotL[k];
			}
			if (k == nPoint) 
			{
				pstar = (1 / sw) * swp;
				qstar = 1 / sw * swq;

				// Calc miu_s
				miu_s = 0;
				for (k = 0; k < nPoint; k++) 
				{
					if (i == oldDotL[k].x && j == oldDotL[k].y) continue;

					Pi = oldDotL[k] - pstar;
					miu_s += w[k] * Pi.dot(Pi);
				}

				curV -= pstar;
				curVJ.x = -curV.y, curVJ.y = curV.x;

				for (k = 0; k < nPoint; k++) 
				{
					if (i == oldDotL[k].x && j == oldDotL[k].y) continue;

					Pi = oldDotL[k] - pstar;
					PiJ.x = -Pi.y, PiJ.y = Pi.x;

					tmpP.x = Pi.dot(curV) * newDotL[k].x -
						PiJ.dot(curV) * newDotL[k].y;
					tmpP.y = -Pi.dot(curVJ) * newDotL[k].x +
						PiJ.dot(curVJ) * newDotL[k].y;
					tmpP *= w[k] / miu_s;
					newP += tmpP;
				}
				newP += qstar;
			}
			else 
			{
				newP = newDotL[k];
			}

			rDx(j, i) = newP.x - i;
			rDy(j, i) = newP.y - j;
		}
	}

}


#endif  // IMGTRANS_MLS_SIMILARITY_H
