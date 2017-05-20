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
#ifndef IMGTRANS_MLS_RIGID_H
#define IMGTRANS_MLS_RIGID_H

#include <vector>
#include "opencv2/opencv.hpp"
#include "imgwarp_mls.h"

//! The class for MLS Rigid transform.
/*!
 * It will try to keep the image rigid. You can set preScale if you
 * can accept uniform transform.
 */
template <typename Type = float>
class ImgWarp_MLS_Rigid : public ImgWarp_MLS<Type>
{
public:
    //! Whether do unify scale on the points before deformation
    bool preScale = false;
    
	ImgWarp_MLS_Rigid();
	Type calcArea(const std::vector<cv::Point_<Type> > &V);
	void calcDelta();

};

//-------------------------------------------------------------------
//Type define 
using ImgWarpRigid = ImgWarp_MLS_Rigid<float>;

//-------------------------------------------------------------------
//Template Class implementation

template <typename Type = float>
ImgWarp_MLS_Rigid<Type>::ImgWarp_MLS_Rigid(){
	VirtualCalcDelta = [this]{this->calcDelta(); };
};

template <typename Type = float>
Type ImgWarp_MLS_Rigid<Type>::calcArea(const std::vector<cv::Point_<Type> > &V)
{
	cv::Point_<Type> lt, rb;
	lt.x = lt.y = 1e10;
	rb.x = rb.y = -1e10;
	for (std::vector<cv::Point_<Type> >::const_iterator i = V.begin(); i != V.end();
		i++) {
		if (i->x < lt.x) lt.x = i->x;
		if (i->x > rb.x) rb.x = i->x;
		if (i->y < lt.y) lt.y = i->y;
		if (i->y > rb.y) rb.y = i->y;
	}
	return (rb.x - lt.x) * (rb.y - lt.y);
}

template <typename Type = float>
void ImgWarp_MLS_Rigid<Type>::calcDelta()
{
	int i, j, k;

	cv::Point_<Type> swq, qstar, newP, tmpP;
	Type sw;

	Type ratio;

	if (preScale) {
		ratio = sqrt(calcArea(newDotL) / calcArea(oldDotL));
		for (i = 0; i < nPoint; i++) newDotL[i] *= 1 / ratio;
	}

	std::vector<Type> w(nPoint);

	rDx.create(tarH, tarW);
	rDy.create(tarH, tarW);

	if (nPoint < 2) {
		rDx.setTo(0);
		rDy.setTo(0);
		return;
	}
	cv::Point_<Type> swp, pstar, curV, curVJ, Pi, PiJ, Qi;
	Type miu_r;

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
			sw = 0;
			swp.x = swp.y = 0;
			swq.x = swq.y = 0;
			newP.x = newP.y = 0;
			curV.x = i;
			curV.y = j;
			for (k = 0; k < nPoint; k++) {
				if ((i == oldDotL[k].x) && j == oldDotL[k].y) break;
				if (alpha == 1)
					w[k] = 1 / ((i - oldDotL[k].x) * (i - oldDotL[k].x) +
					(j - oldDotL[k].y) * (j - oldDotL[k].y));
				else
					w[k] = pow((i - oldDotL[k].x) * (i - oldDotL[k].x) +
					(j - oldDotL[k].y) * (j - oldDotL[k].y),
					-alpha);
				sw = sw + w[k];
				swp = swp + w[k] * oldDotL[k];
				swq = swq + w[k] * newDotL[k];
			}
			if (k == nPoint) {
				pstar = (1 / sw) * swp;
				qstar = 1 / sw * swq;

				// Calc miu_r
				Type s1 = 0, s2 = 0;
				for (k = 0; k < nPoint; k++) {
					if (i == oldDotL[k].x && j == oldDotL[k].y) continue;

					Pi = oldDotL[k] - pstar;
					PiJ.x = -Pi.y, PiJ.y = Pi.x;
					Qi = newDotL[k] - qstar;
					s1 += w[k] * Qi.dot(Pi);
					s2 += w[k] * Qi.dot(PiJ);
				}
				miu_r = sqrt(s1 * s1 + s2 * s2);

				curV -= pstar;
				curVJ.x = -curV.y, curVJ.y = curV.x;

				for (k = 0; k < nPoint; k++) {
					if (i == oldDotL[k].x && j == oldDotL[k].y) continue;

					Pi = oldDotL[k] - pstar;
					PiJ.x = -Pi.y, PiJ.y = Pi.x;

					tmpP.x = Pi.dot(curV) * newDotL[k].x -
						PiJ.dot(curV) * newDotL[k].y;
					tmpP.y = -Pi.dot(curVJ) * newDotL[k].x +
						PiJ.dot(curVJ) * newDotL[k].y;
					tmpP *= w[k] / miu_r;
					newP += tmpP;
				}
				newP += qstar;
			}
			else {
				newP = newDotL[k];
			}

			if (preScale) {
				rDx(j, i) = newP.x * ratio - i;
				rDy(j, i) = newP.y * ratio - j;
			}
			else {
				rDx(j, i) = newP.x - i;
				rDy(j, i) = newP.y - j;
			}
		}
	}

	if (preScale) {
		for (i = 0; i < nPoint; i++) newDotL[i] *= ratio;
	}
}


#endif // IMGTRANS_MLS_RIGID_H
