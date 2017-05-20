#pragma once

#include <thread>
#include <vector>
#include <algorithm>

//-> for(int indice = begin; indice < end; indice++)
//indice : nom de la variable itérative
//begin : debut de l'itération
//end : fin de l'itération
//nth : nombre de thread
void MultiThreadFor(int begin, int end, int nth, std::function<void(int, int)> fFunc)
{
	int iNbTh = std::min(end - begin, nth);
	if (iNbTh > 1)
	{
		std::vector <std::thread> vThread(iNbTh);
		int iSubSize = (end - begin) / iNbTh;
		int iRes = (end - begin) % iNbTh;
		int iStart = begin;
		int iEnd = iStart + iSubSize + iRes;

		for (int ith = 0; ith < iNbTh; ith++)
		{
			vThread[ith] = (std::thread([&, ith](int a, int b)
			{
				for (int indice = a; indice < b; indice++)
					fFunc(indice, ith);
			}, iStart, iEnd));

			iStart = iEnd;
			iEnd += iSubSize;
		}
		for (auto & th : vThread) th.join();
	}
	else
	{
		for (int i = begin; i < end; i++)
			fFunc(i, 0);
	}
}