/************************************************************************************
Modified from Oculus Sample D3D11 application/Window setup functionality for RoomTiny
Author      :   Tom Heath
Copyright   :   Copyright 2014 Oculus, Inc. All Rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*************************************************************************************/
// Include the Oculus SDK
#include "OVR_CAPI_D3D.h"

#include <cstdint>
#include <vector>
#include "d3dcompiler.h"
#include "d3d11.h"
#include "stdio.h"
#include <new>
#if _MSC_VER > 1600
	#include "DirectXMath.h"
	using namespace DirectX;
#else
	#include "xnamath.h"
#endif

#include "opencv2\highgui.hpp"
#include "opencv2\core\directx.hpp"
#include "opencv2\imgproc.hpp"
#include <stdlib.h>

#include "VRdisplay.h"

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

#ifndef VALIDATE
#define VALIDATE(x, msg) if (!(x)) { MessageBoxA(NULL, (msg), "VR Display", MB_ICONERROR | MB_OK); exit(-1); }
#endif

// clean up member COM pointers
template<typename T> void Release(T *&obj)
{
	if (!obj) return;
	obj->Release();
	obj = nullptr;
}

//------------------------------------------------------------
struct DepthBuffer
{
	ID3D11DepthStencilView * TexDsv;

	DepthBuffer(ID3D11Device * Device, int sizeW, int sizeH, int sampleCount = 1)
	{
		DXGI_FORMAT format = DXGI_FORMAT_D32_FLOAT;
		D3D11_TEXTURE2D_DESC dsDesc;
		dsDesc.Width = sizeW;
		dsDesc.Height = sizeH;
		dsDesc.MipLevels = 1;
		dsDesc.ArraySize = 1;
		dsDesc.Format = format;
		dsDesc.SampleDesc.Count = sampleCount;
		dsDesc.SampleDesc.Quality = 0;
		dsDesc.Usage = D3D11_USAGE_DEFAULT;
		dsDesc.CPUAccessFlags = 0;
		dsDesc.MiscFlags = 0;
		dsDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
		ID3D11Texture2D * Tex;
		Device->CreateTexture2D(&dsDesc, NULL, &Tex);
		Device->CreateDepthStencilView(Tex, NULL, &TexDsv);
		Tex->Release();
	}
	~DepthBuffer()
	{
		Release(TexDsv);
	}
};

//----------------------------------------------------------------
struct DataBuffer
{
	ID3D11Buffer * D3DBuffer;
	size_t         Size;

	DataBuffer(ID3D11Device * Device, D3D11_BIND_FLAG use, const void* buffer, size_t size) : Size(size)
	{
		D3D11_BUFFER_DESC desc;   memset(&desc, 0, sizeof(desc));
		desc.Usage = D3D11_USAGE_DYNAMIC;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		desc.BindFlags = use;
		desc.ByteWidth = (unsigned)size;
		D3D11_SUBRESOURCE_DATA sr;
		sr.pSysMem = buffer;
		sr.SysMemPitch = sr.SysMemSlicePitch = 0;
		Device->CreateBuffer(&desc, buffer ? &sr : NULL, &D3DBuffer);
	}
	~DataBuffer()
	{
		Release(D3DBuffer);
	}
};

//---------------------------------------------------------------------
struct DirectX11
{
	HWND                     Window;
	bool                     Running;
	bool                     Key[256];
	int                      WinSizeW;
	int                      WinSizeH;
	ID3D11Device           * Device;
	ID3D11DeviceContext    * Context;
	IDXGISwapChain         * SwapChain;
	DepthBuffer            * MainDepthBuffer;
	ID3D11Texture2D        * BackBuffer;
	ID3D11RenderTargetView * BackBufferRT;
	// Fixed size buffer for shader constants, before copied into buffer
	static const int         UNIFORM_DATA_SIZE = 2000;
	unsigned char            UniformData[UNIFORM_DATA_SIZE];
	DataBuffer             * UniformBufferGen;
	HINSTANCE                hInstance;

	static LRESULT CALLBACK WindowProc(_In_ HWND hWnd, _In_ UINT Msg, _In_ WPARAM wParam, _In_ LPARAM lParam)
	{
		auto p = reinterpret_cast<DirectX11 *>(GetWindowLongPtr(hWnd, 0));
		switch (Msg)
		{
		case WM_KEYDOWN:
			p->Key[wParam] = true;
			break;
		case WM_KEYUP:
			p->Key[wParam] = false;
			break;
		case WM_DESTROY:
			p->Running = false;
			break;
		default:
			return DefWindowProcW(hWnd, Msg, wParam, lParam);
		}
		if ((p->Key['Q'] && p->Key[VK_CONTROL]) || p->Key[VK_ESCAPE])
		{
			p->Running = false;
		}
		return 0;
	}

	DirectX11() :
		Window(nullptr),
		Running(false),
		WinSizeW(0),
		WinSizeH(0),
		Device(nullptr),
		Context(nullptr),
		SwapChain(nullptr),
		MainDepthBuffer(nullptr),
		BackBuffer(nullptr),
		BackBufferRT(nullptr),
		UniformBufferGen(nullptr),
		hInstance(nullptr)
	{
		// Clear input
		for (int i = 0; i < sizeof(Key) / sizeof(Key[0]); ++i)
			Key[i] = false;
	}

	~DirectX11()
	{
		ReleaseDevice();
		CloseWindow();
	}

	bool InitWindow(HINSTANCE hinst, LPCWSTR title)
	{
		hInstance = hinst;
		Running = true;

		WNDCLASSW wc;
		memset(&wc, 0, sizeof(wc));
		wc.lpszClassName = L"App";
		wc.style = CS_OWNDC;
		wc.lpfnWndProc = WindowProc;
		wc.cbWndExtra = sizeof(this);
		RegisterClassW(&wc);

		// adjust the window size and show at InitDevice time
		Window = CreateWindowW(wc.lpszClassName, title, WS_OVERLAPPEDWINDOW, 0, 0, 0, 0, 0, 0, hinst, 0);
		if (!Window) return false;

		SetWindowLongPtr(Window, 0, LONG_PTR(this));

		return true;
	}

	void CloseWindow()
	{
		if (Window)
		{
			DestroyWindow(Window);
			Window = nullptr;
			UnregisterClassW(L"App", hInstance);
		}
	}

	bool InitDevice(int vpW, int vpH, const LUID* pLuid, bool windowed = true, int scale = 1)
	{
		WinSizeW = vpW;
		WinSizeH = vpH;

		if (scale == 0)
			scale = 1;

		RECT size = { 0, 0, vpW / scale, vpH / scale };
		AdjustWindowRect(&size, WS_OVERLAPPEDWINDOW, false);
		const UINT flags = SWP_NOMOVE | SWP_NOZORDER | SWP_SHOWWINDOW;
		if (!SetWindowPos(Window, nullptr, 0, 0, size.right - size.left, size.bottom - size.top, flags))
			return false;

		IDXGIFactory * DXGIFactory = nullptr;
		HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory), (void**)(&DXGIFactory));
		VALIDATE((hr == ERROR_SUCCESS), "CreateDXGIFactory1 failed");

		IDXGIAdapter * Adapter = nullptr;
		for (UINT iAdapter = 0; DXGIFactory->EnumAdapters(iAdapter, &Adapter) != DXGI_ERROR_NOT_FOUND; ++iAdapter)
		{
			DXGI_ADAPTER_DESC adapterDesc;
			Adapter->GetDesc(&adapterDesc);
			if ((pLuid == nullptr) || memcmp(&adapterDesc.AdapterLuid, pLuid, sizeof(LUID)) == 0)
				break;
			Release(Adapter);
		}

		auto DriverType = Adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;
		hr = D3D11CreateDevice(Adapter, DriverType, 0, 0, 0, 0, D3D11_SDK_VERSION, &Device, 0, &Context);
		Release(Adapter);
		VALIDATE((hr == ERROR_SUCCESS), "D3D11CreateDevice failed");

		// Create swap chain
		DXGI_SWAP_CHAIN_DESC scDesc;
		memset(&scDesc, 0, sizeof(scDesc));
		scDesc.BufferCount = 2;
		scDesc.BufferDesc.Width = WinSizeW;
		scDesc.BufferDesc.Height = WinSizeH;
		scDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		scDesc.BufferDesc.RefreshRate.Denominator = 1;
		scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		scDesc.OutputWindow = Window;
		scDesc.SampleDesc.Count = 1;
		scDesc.Windowed = windowed;
		scDesc.SwapEffect = DXGI_SWAP_EFFECT_SEQUENTIAL;
		hr = DXGIFactory->CreateSwapChain(Device, &scDesc, &SwapChain);
		Release(DXGIFactory);
		VALIDATE((hr == ERROR_SUCCESS), "CreateSwapChain failed");

		// Create backbuffer
		SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&BackBuffer);
		hr = Device->CreateRenderTargetView(BackBuffer, NULL, &BackBufferRT);
		VALIDATE((hr == ERROR_SUCCESS), "CreateRenderTargetView failed");

		// Main depth buffer
		MainDepthBuffer = new DepthBuffer(Device, WinSizeW, WinSizeH);
		Context->OMSetRenderTargets(1, &BackBufferRT, MainDepthBuffer->TexDsv);

		// Buffer for shader constants
		UniformBufferGen = new DataBuffer(Device, D3D11_BIND_CONSTANT_BUFFER, NULL, UNIFORM_DATA_SIZE);
		Context->VSSetConstantBuffers(0, 1, &UniformBufferGen->D3DBuffer);

		// Set max frame latency to 1
		IDXGIDevice1* DXGIDevice1 = nullptr;
		hr = Device->QueryInterface(__uuidof(IDXGIDevice1), (void**)&DXGIDevice1);
		VALIDATE((hr == ERROR_SUCCESS), "QueryInterface failed");
		DXGIDevice1->SetMaximumFrameLatency(1);
		Release(DXGIDevice1);

		return true;
	}

	void SetAndClearRenderTarget(ID3D11RenderTargetView * rendertarget, struct DepthBuffer * depthbuffer, float R = 0, float G = 0, float B = 0, float A = 0)
	{
		float black[] = { R, G, B, A }; // Important that alpha=0, if want pixels to be transparent, for manual layers
		Context->OMSetRenderTargets(1, &rendertarget, (depthbuffer ? depthbuffer->TexDsv : nullptr));
		Context->ClearRenderTargetView(rendertarget, black);
		if (depthbuffer)
			Context->ClearDepthStencilView(depthbuffer->TexDsv, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1, 0);
	}

	void SetViewport(float vpX, float vpY, float vpW, float vpH)
	{
		D3D11_VIEWPORT D3Dvp;
		D3Dvp.Width = vpW;    D3Dvp.Height = vpH;
		D3Dvp.MinDepth = 0;   D3Dvp.MaxDepth = 1;
		D3Dvp.TopLeftX = vpX; D3Dvp.TopLeftY = vpY;
		Context->RSSetViewports(1, &D3Dvp);
	}

	bool HandleMessages(void)
	{
		MSG msg;
		while (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		// This is to provide a means to terminate after a maximum number of frames
		// to facilitate automated testing
#ifdef MAX_FRAMES_ACTIVE 
		if (maxFrames > 0)
		{
			if (--maxFrames <= 0)
				Running = false;
		}
#endif
		return Running;
	}

	void Run(bool(*MainLoop)(bool retryCreate))
	{
		// false => just fail on any error
		VALIDATE(MainLoop(false), "Oculus Rift not detected.");
		while (HandleMessages())
		{
			// true => we'll attempt to retry for ovrError_DisplayLost
			if (!MainLoop(true))
				break;
			// Sleep a bit before retrying to reduce CPU load while the HMD is disconnected
			Sleep(10);
		}
	}

	void ReleaseDevice()
	{
		Release(BackBuffer);
		Release(BackBufferRT);
		if (SwapChain)
		{
			SwapChain->SetFullscreenState(FALSE, NULL);
			Release(SwapChain);
		}
		Release(Context);
		Release(Device);
		delete MainDepthBuffer;
		MainDepthBuffer = nullptr;
		delete UniformBufferGen;
		UniformBufferGen = nullptr;
	}
};

// global DX11 state
static struct DirectX11 DIRECTX;

//------------------------------------------------------------
//Image on a material
struct Texture
{
	ID3D11Texture2D            * Tex;
	ID3D11ShaderResourceView   * TexSv;
	ID3D11RenderTargetView     * TexRtv;
	int                          SizeW, SizeH, MipLevels;

	Texture() : Tex(nullptr), TexSv(nullptr), TexRtv(nullptr) {};
	void Init(int sizeW, int sizeH, bool rendertarget, int mipLevels, int sampleCount)
	{
		SizeW = sizeW;
		SizeH = sizeH;
		MipLevels = mipLevels;

		D3D11_TEXTURE2D_DESC dsDesc;
		dsDesc.Width = SizeW;
		dsDesc.Height = SizeH;
		dsDesc.MipLevels = MipLevels;
		dsDesc.ArraySize = 1;
		dsDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		dsDesc.SampleDesc.Count = sampleCount;
		dsDesc.SampleDesc.Quality = 0;
		dsDesc.Usage = D3D11_USAGE_DEFAULT;
		dsDesc.CPUAccessFlags = 0;
		dsDesc.MiscFlags = 0;
		dsDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		if (rendertarget) dsDesc.BindFlags |= D3D11_BIND_RENDER_TARGET;

		DIRECTX.Device->CreateTexture2D(&dsDesc, NULL, &Tex);
		DIRECTX.Device->CreateShaderResourceView(Tex, NULL, &TexSv);
		TexRtv = nullptr;
		if (rendertarget) DIRECTX.Device->CreateRenderTargetView(Tex, NULL, &TexRtv);
	}
	Texture(int sizeW, int sizeH, bool rendertarget, int mipLevels = 1, int sampleCount = 1)
	{
		Init(sizeW, sizeH, rendertarget, mipLevels, sampleCount);
	}
	Texture(bool rendertarget, int sizeW, int sizeH, int autoFillData = 0, int sampleCount = 1)
	{
		Init(sizeW, sizeH, rendertarget, autoFillData ? 8 : 1, sampleCount);
	}

	~Texture()
	{
		Release(TexRtv);
		Release(TexSv);
		Release(Tex);
	}

	/*void FillTexture(uint32_t * pix)
	{
		//Make local ones, because will be reducing them
		int sizeW = SizeW;
		int sizeH = SizeH;
		for (int level = 0; level < MipLevels; level++)
		{
			DIRECTX.Context->UpdateSubresource(Tex, level, NULL, (unsigned char *)pix, sizeW * 4, sizeH * sizeW * 4);

			for (int j = 0; j < (sizeH & ~1); j += 2)
			{
				uint8_t* psrc = (uint8_t *)pix + (sizeW * j * 4);
				uint8_t* pdest = (uint8_t *)pix + (sizeW * j);
				for (int i = 0; i < sizeW >> 1; i++, psrc += 8, pdest += 4)
				{
					pdest[0] = (((int)psrc[0]) + psrc[4] + psrc[sizeW * 4 + 0] + psrc[sizeW * 4 + 4]) >> 2;
					pdest[1] = (((int)psrc[1]) + psrc[5] + psrc[sizeW * 4 + 1] + psrc[sizeW * 4 + 5]) >> 2;
					pdest[2] = (((int)psrc[2]) + psrc[6] + psrc[sizeW * 4 + 2] + psrc[sizeW * 4 + 6]) >> 2;
					pdest[3] = (((int)psrc[3]) + psrc[7] + psrc[sizeW * 4 + 3] + psrc[sizeW * 4 + 7]) >> 2;
				}
			}
			sizeW >>= 1;  sizeH >>= 1;
		}
	}*/

	void SetTextureMat(cv::Mat input){
		DIRECTX.Context->UpdateSubresource(Tex, 0, NULL, (unsigned int *)input.data, SizeW * 4, SizeH * SizeW * 4);
	}
};

//-----------------------------------------------------
//Manages GPU shaders and other GPU specific things
//Holds the texture
struct Material
{
	ID3D11VertexShader      * VertexShader, *VertexShaderInstanced;
	ID3D11PixelShader       * PixelShader;
	Texture                 * Tex;
	ID3D11InputLayout       * InputLayout;
	UINT                      VertexSize;
	ID3D11SamplerState      * SamplerState;
	ID3D11RasterizerState   * Rasterizer;
	ID3D11DepthStencilState * DepthState;
	ID3D11BlendState        * BlendState;

	enum { MAT_WRAP = 1, MAT_WIRE = 2, MAT_ZALWAYS = 4, MAT_NOCULL = 8, MAT_TRANS = 16 };
	Material(Texture * t, uint32_t flags = MAT_WRAP | MAT_TRANS, D3D11_INPUT_ELEMENT_DESC * vertexDesc = NULL, int numVertexDesc = 3,
		char* vertexShader = NULL, char* pixelShader = NULL, int vSize = 24) : Tex(t), VertexSize(vSize)
	{
		D3D11_INPUT_ELEMENT_DESC defaultVertexDesc[] = {
			{ "Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "Color", 0, DXGI_FORMAT_B8G8R8A8_UNORM, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 }, };

		// Use defaults if no shaders specified
		char* defaultVertexShaderSrc =
			"float4x4 ProjView;  float4 MasterCol;"
			"void main(in  float4 Position  : POSITION,    in  float4 Color : COLOR0, in  float2 TexCoord  : TEXCOORD0,"
			"          out float4 oPosition : SV_Position, out float4 oColor: COLOR0, out float2 oTexCoord : TEXCOORD0)"
			"{   oPosition = mul(ProjView, Position); oTexCoord = TexCoord; "
			"    oColor = MasterCol * Color; }";
		char* defaultPixelShaderSrc =
			"Texture2D Texture   : register(t0); SamplerState Linear : register(s0); "
			"float4 main(in float4 Position : SV_Position, in float4 Color: COLOR0, in float2 TexCoord : TEXCOORD0) : SV_Target"
			"{   float4 TexCol = Texture.Sample(Linear, TexCoord); "
			"    if (TexCol.a==0) clip(-1); " // If alpha = 0, don't draw
			"    return(Color * TexCol); }";

		// vertex shader for instanced stereo
		char* instancedStereoVertexShaderSrc =
			"float4x4 modelViewProj[2];  float4 MasterCol;"
			"void main(in  float4 Position  : POSITION,    in  float4 Color : COLOR0, in  float2 TexCoord  : TEXCOORD0,"
			"          in  uint instanceID : SV_InstanceID, "
			"          out float4 oPosition : SV_Position, out float4 oColor: COLOR0, out float2 oTexCoord : TEXCOORD0,"
			"          out float oClipDist : SV_ClipDistance0, out float oCullDist : SV_CullDistance0)"
			"{"
			"   const float4 EyeClipPlane[2] = { { -1, 0, 0, 0 }, { 1, 0, 0, 0 } };"
			"   uint eyeIndex = instanceID & 1;"
			// transform to clip space for correct eye (includes offset and scale)
			"   oPosition = mul(modelViewProj[eyeIndex], Position); "
			// calculate distance from left/right clip plane (try setting to 0 to see why clipping is necessary)
			"   oCullDist = oClipDist = dot(EyeClipPlane[eyeIndex], oPosition);"
			"   oTexCoord = TexCoord; "
			"   oColor = MasterCol * Color;"
			"}";

		if (!vertexDesc)   vertexDesc = defaultVertexDesc;
		if (!vertexShader) vertexShader = defaultVertexShaderSrc;
		if (!pixelShader)  pixelShader = defaultPixelShaderSrc;

		// Create vertex shader
		ID3DBlob * blobData;
		ID3DBlob * errorBlob = nullptr;
		HRESULT result = D3DCompile(vertexShader, strlen(vertexShader), 0, 0, 0, "main", "vs_4_0", 0, 0, &blobData, &errorBlob);
		if (FAILED(result))
		{
			MessageBoxA(NULL, (char *)errorBlob->GetBufferPointer(), "Error compiling vertex shader", MB_OK);
			exit(-1);
		}
		DIRECTX.Device->CreateVertexShader(blobData->GetBufferPointer(), blobData->GetBufferSize(), NULL, &VertexShader);

		// Create input layout
		DIRECTX.Device->CreateInputLayout(vertexDesc, numVertexDesc,
			blobData->GetBufferPointer(), blobData->GetBufferSize(), &InputLayout);
		blobData->Release();

		// Create vertex shader for instancing
		result = D3DCompile(instancedStereoVertexShaderSrc, strlen(instancedStereoVertexShaderSrc), 0, 0, 0, "main", "vs_4_0", 0, 0, &blobData, &errorBlob);
		if (FAILED(result))
		{
			MessageBoxA(NULL, (char *)errorBlob->GetBufferPointer(), "Error compiling vertex shader", MB_OK);
			exit(-1);
		}
		DIRECTX.Device->CreateVertexShader(blobData->GetBufferPointer(), blobData->GetBufferSize(), NULL, &VertexShaderInstanced);
		blobData->Release();

		// Create pixel shader
		D3DCompile(pixelShader, strlen(pixelShader), 0, 0, 0, "main", "ps_4_0", 0, 0, &blobData, 0);
		DIRECTX.Device->CreatePixelShader(blobData->GetBufferPointer(), blobData->GetBufferSize(), NULL, &PixelShader);
		blobData->Release();

		// Create sampler state
		D3D11_SAMPLER_DESC ss; memset(&ss, 0, sizeof(ss));
		ss.AddressU = ss.AddressV = ss.AddressW = flags & MAT_WRAP ? D3D11_TEXTURE_ADDRESS_WRAP : D3D11_TEXTURE_ADDRESS_BORDER;
		ss.Filter = D3D11_FILTER_ANISOTROPIC;
		ss.MaxAnisotropy = 8;
		ss.MaxLOD = 15;
		DIRECTX.Device->CreateSamplerState(&ss, &SamplerState);

		// Create rasterizer
		D3D11_RASTERIZER_DESC rs; memset(&rs, 0, sizeof(rs));
		rs.AntialiasedLineEnable = rs.DepthClipEnable = true;
		rs.CullMode = flags & MAT_NOCULL ? D3D11_CULL_NONE : D3D11_CULL_BACK;
		rs.FillMode = flags & MAT_WIRE ? D3D11_FILL_WIREFRAME : D3D11_FILL_SOLID;
		DIRECTX.Device->CreateRasterizerState(&rs, &Rasterizer);

		// Create depth state
		D3D11_DEPTH_STENCIL_DESC dss;
		memset(&dss, 0, sizeof(dss));
		dss.DepthEnable = false;//true;
		dss.DepthFunc = flags & MAT_ZALWAYS ? D3D11_COMPARISON_ALWAYS : D3D11_COMPARISON_LESS;
		dss.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
		DIRECTX.Device->CreateDepthStencilState(&dss, &DepthState);

		//Create blend state - trans or otherwise
		D3D11_BLEND_DESC bm;
		memset(&bm, 0, sizeof(bm));
		bm.RenderTarget[0].BlendEnable = flags & MAT_TRANS ? true : false;
		bm.RenderTarget[0].BlendOp = bm.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
		bm.RenderTarget[0].SrcBlend = bm.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
		bm.RenderTarget[0].DestBlend = bm.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
		bm.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
		DIRECTX.Device->CreateBlendState(&bm, &BlendState);
	}
	~Material()
	{
		Release(VertexShader);
		Release(VertexShaderInstanced);
		Release(PixelShader);
		delete Tex; Tex = nullptr;
		Release(InputLayout);
		Release(SamplerState);
		Release(Rasterizer);
		Release(DepthState);
		Release(BlendState);
	}
};

//----------------------------------------------------------------------
struct Vertex
{
	XMFLOAT3  Pos;
	uint32_t  C;
	float     U, V;
	Vertex() {};
	Vertex(XMFLOAT3 pos, uint32_t c, float u, float v) : Pos(pos), C(c), U(u), V(v) {};
};

//-----------------------------------------------------------------------
//Manages vertexs
struct TriangleSet
{
	int       numVertices, numIndices, maxBuffer;
	Vertex    * Vertices;
	short     * Indices;
	TriangleSet(int maxTriangles = 2000) : maxBuffer(3 * maxTriangles)
	{
		numVertices = numIndices = 0;
		Vertices = (Vertex *)_aligned_malloc(maxBuffer *sizeof(Vertex), 16);
		Indices = (short *)_aligned_malloc(maxBuffer *sizeof(short), 16);
	}
	~TriangleSet()
	{
		_aligned_free(Vertices);
		_aligned_free(Indices);
	}
	void AddQuad(Vertex v0, Vertex v1, Vertex v2, Vertex v3) { AddTriangle(v0, v1, v2);	AddTriangle(v3, v2, v1); }
	void AddTriangle(Vertex v0, Vertex v1, Vertex v2)
	{
		VALIDATE(numVertices <= (maxBuffer - 3), "Insufficient triangle set");
		for (int i = 0; i < 3; i++) Indices[numIndices++] = numVertices + i;
		Vertices[numVertices++] = v0;
		Vertices[numVertices++] = v1;
		Vertices[numVertices++] = v2;
	}

	uint32_t ModifyColor(uint32_t c, XMFLOAT3 pos)
	{
#define GetLengthLocal(v)  (sqrt(v.x*v.x + v.y*v.y + v.z*v.z))
		float dist1 = GetLengthLocal(XMFLOAT3(pos.x - (-2), pos.y - (4), pos.z - (-2)));
		float dist2 = GetLengthLocal(XMFLOAT3(pos.x - (3), pos.y - (4), pos.z - (-3)));
		float dist3 = GetLengthLocal(XMFLOAT3(pos.x - (-4), pos.y - (3), pos.z - (25)));
		int   bri = rand() % 160;
		float R = ((c >> 16) & 0xff) * (bri + 192.0f*(0.65f + 8 / dist1 + 1 / dist2 + 4 / dist3)) / 255.0f;
		float G = ((c >> 8) & 0xff) * (bri + 192.0f*(0.65f + 8 / dist1 + 1 / dist2 + 4 / dist3)) / 255.0f;
		float B = ((c >> 0) & 0xff) * (bri + 192.0f*(0.65f + 8 / dist1 + 1 / dist2 + 4 / dist3)) / 255.0f;
		return((c & 0xff000000) + ((R>255 ? 255 : (uint32_t)R) << 16) + ((G>255 ? 255 : (uint32_t)G) << 8) + (B>255 ? 255 : (uint32_t)B));
	}

	void AddSingleQuad(float x1, float y1, float z1, float x2, float y2, float z2, uint32_t c)
	{
		AddQuad(Vertex(XMFLOAT3(x2, y1, z2), ModifyColor(c, XMFLOAT3(x2, y1, z2)), x2, y1),
			Vertex(XMFLOAT3(x1, y1, z2), ModifyColor(c, XMFLOAT3(x1, y1, z2)), x1, y1),
			Vertex(XMFLOAT3(x2, y2, z2), ModifyColor(c, XMFLOAT3(x2, y2, z2)), x2, y2),
			Vertex(XMFLOAT3(x1, y2, z2), ModifyColor(c, XMFLOAT3(x1, y2, z2)), x1, y2));
	}
};

//----------------------------------------------------------------------
//A general object on screen. Has a materal and a TrangleSet. Uses those to create final render of object
struct Model
{
	XMFLOAT3     Pos;
	XMFLOAT4     Rot;
	Material   * Fill;
	DataBuffer * VertexBuffer;
	DataBuffer * IndexBuffer;
	int          NumIndices;

	Model() : Fill(nullptr), VertexBuffer(nullptr), IndexBuffer(nullptr) {};
	void Init(TriangleSet * t)
	{
		NumIndices = t->numIndices;
		VertexBuffer = new DataBuffer(DIRECTX.Device, D3D11_BIND_VERTEX_BUFFER, &t->Vertices[0], t->numVertices * sizeof(Vertex));
		IndexBuffer = new DataBuffer(DIRECTX.Device, D3D11_BIND_INDEX_BUFFER, &t->Indices[0], t->numIndices * sizeof(short));
	}
	Model(TriangleSet * t, XMFLOAT3 argPos, XMFLOAT4 argRot, Material * argFill) :
		Pos(argPos),
		Rot(argRot),
		Fill(argFill)
	{
		Init(t);
	}
	// 2D scenes, for latency tester and full screen copies, etc
	Model(Material * mat, float minx, float miny, float maxx, float maxy, float zDepth = -1) :
		Pos(XMFLOAT3(0, 0, 0)),
		Rot(XMFLOAT4(0, 0, 0, 1)),
		Fill(mat)
	{
		TriangleSet quad;
		unsigned int color = 0xffffffff;//sets argb tint
		quad.AddQuad(Vertex(XMFLOAT3(minx, miny, zDepth), color, 0, 1),
			Vertex(XMFLOAT3(minx, maxy, zDepth), color, 0, 0),
			Vertex(XMFLOAT3(maxx, miny, zDepth), color, 1, 1),
			Vertex(XMFLOAT3(maxx, maxy, zDepth), color, 1, 0));
		Init(&quad);
	}
	~Model()
	{
		delete Fill; Fill = nullptr;
		delete VertexBuffer; VertexBuffer = nullptr;
		delete IndexBuffer; IndexBuffer = nullptr;
	}

	void Render(XMMATRIX * projView, float R, float G, float B, float A, bool standardUniforms)
	{
		XMMATRIX modelMat = XMMatrixMultiply(XMMatrixRotationQuaternion(XMLoadFloat4(&Rot)), XMMatrixTranslationFromVector(XMLoadFloat3(&Pos)));
		XMMATRIX mat = XMMatrixMultiply(modelMat, *projView);
		float col[] = { R, G, B, A };
		if (standardUniforms) memcpy(DIRECTX.UniformData + 0, &mat, 64); // ProjView
		if (standardUniforms) memcpy(DIRECTX.UniformData + 64, &col, 16); // MasterCol
		D3D11_MAPPED_SUBRESOURCE map;
		DIRECTX.Context->Map(DIRECTX.UniformBufferGen->D3DBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map);
		memcpy(map.pData, &DIRECTX.UniformData, DIRECTX.UNIFORM_DATA_SIZE);
		DIRECTX.Context->Unmap(DIRECTX.UniformBufferGen->D3DBuffer, 0);
		DIRECTX.Context->IASetInputLayout(Fill->InputLayout);
		DIRECTX.Context->IASetIndexBuffer(IndexBuffer->D3DBuffer, DXGI_FORMAT_R16_UINT, 0);
		UINT offset = 0;
		DIRECTX.Context->IASetVertexBuffers(0, 1, &VertexBuffer->D3DBuffer, &Fill->VertexSize, &offset);
		DIRECTX.Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		DIRECTX.Context->VSSetShader(Fill->VertexShader, NULL, 0);
		DIRECTX.Context->PSSetShader(Fill->PixelShader, NULL, 0);
		DIRECTX.Context->PSSetSamplers(0, 1, &Fill->SamplerState);
		DIRECTX.Context->RSSetState(Fill->Rasterizer);
		DIRECTX.Context->OMSetDepthStencilState(Fill->DepthState, 0);
		DIRECTX.Context->OMSetBlendState(Fill->BlendState, NULL, 0xffffffff);
		DIRECTX.Context->PSSetShaderResources(0, 1, &Fill->Tex->TexSv);

		DIRECTX.Context->DrawIndexed((UINT)NumIndices, 0, 0);
	}

};

//------------------------------------------------------------
// ovrSwapTextureSet wrapper class that also maintains the render target views
// needed for D3D11 rendering.
struct OculusTexture
{
	ovrSession               Session;
	ovrTextureSwapChain      TextureChain;
	std::vector<ID3D11RenderTargetView*> TexRtv;

	OculusTexture() :
		Session(nullptr),
		TextureChain(nullptr)
	{
	}

	bool Init(ovrSession session, int sizeW, int sizeH)
	{
		Session = session;

		ovrTextureSwapChainDesc desc = {};
		desc.Type = ovrTexture_2D;
		desc.ArraySize = 1;
		desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
		desc.Width = sizeW;
		desc.Height = sizeH;
		desc.MipLevels = 1;
		desc.SampleCount = 1;
		desc.MiscFlags = ovrTextureMisc_DX_Typeless;
		desc.BindFlags = ovrTextureBind_DX_RenderTarget;
		desc.StaticImage = ovrFalse;

		ovrResult result = ovr_CreateTextureSwapChainDX(session, DIRECTX.Device, &desc, &TextureChain);
		if (!OVR_SUCCESS(result))
			return false;

		int textureCount = 0;
		ovr_GetTextureSwapChainLength(Session, TextureChain, &textureCount);
		for (int i = 0; i < textureCount; ++i)
		{
			ID3D11Texture2D* tex = nullptr;
			ovr_GetTextureSwapChainBufferDX(Session, TextureChain, i, IID_PPV_ARGS(&tex));
			D3D11_RENDER_TARGET_VIEW_DESC rtvd = {};
			rtvd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			rtvd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
			ID3D11RenderTargetView* rtv;
			DIRECTX.Device->CreateRenderTargetView(tex, &rtvd, &rtv);
			TexRtv.push_back(rtv);
			tex->Release();
		}

		return true;
	}

	~OculusTexture()
	{
		for (int i = 0; i < (int)TexRtv.size(); ++i)
		{
			Release(TexRtv[i]);
		}
		if (TextureChain)
		{
			ovr_DestroyTextureSwapChain(Session, TextureChain);
		}
	}

	ID3D11RenderTargetView* GetRTV()
	{
		int index = 0;
		ovr_GetTextureSwapChainCurrentIndex(Session, TextureChain, &index);
		return TexRtv[index];
	}

	// Commit changes
	void Commit()
	{
		ovr_CommitTextureSwapChain(Session, TextureChain);
	}
};


Model* screen;
int texture_width;
int texture_height;

ovrSession session;
ovrHmdDesc hmdDesc;
// Initialize these to nullptr here to handle device lost failures cleanly
OculusTexture  * pEyeRenderTexture[2] = { nullptr, nullptr };
DepthBuffer    * pEyeDepthBuffer[2] = { nullptr, nullptr };
ovrMirrorTexture mirrorTexture = nullptr;
ovrRecti         eyeRenderViewport[2];// Make the eye render buffers (caution if actual size < requested due to HW limits). 
ovrResult		 result;

bool isVisible = true;
long long frameIndex = 0;

// return true for success, false for any failure
bool SetupMainLoop()
{
	ovrGraphicsLuid luid;
	result = ovr_Create(&session, &luid);
	if (!OVR_SUCCESS(result))
		return false;

	hmdDesc = ovr_GetHmdDesc(session);

	// Setup Device and Graphics
	// Note: the mirror window can be any size, for this sample we use 1/2 the HMD resolution
	if (!DIRECTX.InitDevice(hmdDesc.Resolution.w / 2, hmdDesc.Resolution.h / 2, reinterpret_cast<LUID*>(&luid))){
		return false;//TODO should we return a retry notice, if the headset is disconnected?
	}

	for (int eye = 0; eye < 2; ++eye)
	{
		ovrSizei idealSize = ovr_GetFovTextureSize(session, (ovrEyeType)eye, hmdDesc.DefaultEyeFov[eye], 1.0f);
		pEyeRenderTexture[eye] = new OculusTexture();
		if (!pEyeRenderTexture[eye]->Init(session, idealSize.w, idealSize.h))
		{
			MessageBoxA(NULL, ("Failed to create eye texture."), "VR Display", MB_ICONERROR | MB_OK);
			return false;
		}
		pEyeDepthBuffer[eye] = new DepthBuffer(DIRECTX.Device, idealSize.w, idealSize.h);
		eyeRenderViewport[eye].Pos.x = 0;
		eyeRenderViewport[eye].Pos.y = 0;
		eyeRenderViewport[eye].Size = idealSize;
		if (!pEyeRenderTexture[eye]->TextureChain)
		{
			MessageBoxA(NULL, ("Failed to create texture."), "VR Display", MB_ICONERROR | MB_OK);
			return false;
		}
	}

	// Create a mirror to see on the monitor.
	ovrMirrorTextureDesc mirrorDesc = {};
	mirrorDesc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
	mirrorDesc.Width = DIRECTX.WinSizeW;
	mirrorDesc.Height = DIRECTX.WinSizeH;
	result = ovr_CreateMirrorTextureDX(session, DIRECTX.Device, &mirrorDesc, &mirrorTexture);
	if (!OVR_SUCCESS(result))
	{
		MessageBoxA(NULL, ("Failed to create mirror texture."), "VR Display", MB_ICONERROR | MB_OK);
		return false;
	}

	// Create the output image, a 2d quad directly in front of the screen
	/*
	TriangleSet twod_quad;
	twod_quad.AddSingleQuad(-1, -1, -1, 1, 1, -1, 0xff808080);
	screen = new Model(&twod_quad, XMFLOAT3(0, 0, 0), XMFLOAT4(0, 0, 0, 1),
	new Material(
	new Texture(false, 256, 256, Texture::AUTO_WALL)
	)
	);
	*/
	Texture* tex = new Texture(false, texture_width, texture_height);
	screen = new Model(
		new Material(tex),
		-1, -1, 1, 1
	);

	// FloorLevel will give tracking poses where the floor height is 0
	ovr_SetTrackingOriginType(session, ovrTrackingOrigin_FloorLevel);

	return true;
}

bool Main_VR_Render_Loop(){
	// Call ovr_GetRenderDesc each frame to get the ovrEyeRenderDesc, as the returned values (e.g. HmdToEyeOffset) may change at runtime.
	ovrEyeRenderDesc eyeRenderDesc[2];
	eyeRenderDesc[0] = ovr_GetRenderDesc(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
	eyeRenderDesc[1] = ovr_GetRenderDesc(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);

	// Get both eye poses simultaneously, with IPD offset already included. 
	ovrPosef         EyeRenderPose[2];
	ovrVector3f      HmdToEyeOffset[2] = { eyeRenderDesc[0].HmdToEyeOffset,
										   eyeRenderDesc[1].HmdToEyeOffset };

	double sensorSampleTime;    // sensorSampleTime is fed into the layer later
	ovr_GetEyePoses(session, frameIndex, ovrTrue, HmdToEyeOffset, EyeRenderPose, &sensorSampleTime);

	// Render Scene to Eye Buffers
	if (isVisible)
	{
		for (int eye = 0; eye < 2; ++eye)
		{
			// Clear and set up rendertarget
			DIRECTX.SetAndClearRenderTarget(pEyeRenderTexture[eye]->GetRTV(), pEyeDepthBuffer[eye]);
			DIRECTX.SetViewport((float)eyeRenderViewport[eye].Pos.x, (float)eyeRenderViewport[eye].Pos.y,
				(float)eyeRenderViewport[eye].Size.w, (float)eyeRenderViewport[eye].Size.h);

			//get the matrix projection to the oculus distortion
			ovrMatrix4f p = ovrMatrix4f_Projection(eyeRenderDesc[eye].Fov, 0.2f, 1000.0f, ovrProjection_None);
			XMMATRIX proj = XMMatrixSet(p.M[0][0], p.M[1][0], p.M[2][0], p.M[3][0],
				p.M[0][1], p.M[1][1], p.M[2][1], p.M[3][1],
				p.M[0][2], p.M[1][2], p.M[2][2], p.M[3][2],
				p.M[0][3], p.M[1][3], p.M[2][3], p.M[3][3]);

			//render using the projection
			screen->Render(&proj, 1, 1, 1, 1, true);

			// Commit rendering to the swap chain
			pEyeRenderTexture[eye]->Commit();
		}
	}

	// Initialize our single full screen Fov layer.
	ovrLayerEyeFov ld = {};
	ld.Header.Type = ovrLayerType_EyeFov;
	ld.Header.Flags = 0;// ovrLayerFlag_HeadLocked | ovrLayerFlag_HighQuality;

	for (int eye = 0; eye < 2; ++eye)
	{
		ld.ColorTexture[eye] = pEyeRenderTexture[eye]->TextureChain;
		ld.Viewport[eye] = eyeRenderViewport[eye];
		ld.Fov[eye] = hmdDesc.DefaultEyeFov[eye];
		ld.RenderPose[eye] = EyeRenderPose[eye];
		ld.SensorSampleTime = sensorSampleTime;
	}

	ovrLayerHeader* layers = &ld.Header;
	result = ovr_SubmitFrame(session, frameIndex, nullptr, &layers, 1);
	// exit the rendering loop if submit returns an error, will retry on ovrError_DisplayLost
	if (!OVR_SUCCESS(result)){
		return (result == ovrError_DisplayLost);
	}

	isVisible = (result == ovrSuccess);

	ovrSessionStatus sessionStatus;
	ovr_GetSessionStatus(session, &sessionStatus);
	if (sessionStatus.ShouldQuit){
		return false;
	}
	if (sessionStatus.ShouldRecenter)
		ovr_RecenterTrackingOrigin(session);

	// Render mirror
	ID3D11Texture2D* tex = nullptr;
	ovr_GetMirrorTextureBufferDX(session, mirrorTexture, IID_PPV_ARGS(&tex));
	DIRECTX.Context->CopyResource(DIRECTX.BackBuffer, tex);
	tex->Release();
	DIRECTX.SwapChain->Present(0, 0);

	frameIndex++;

	return true;
}

void Exit_VR(){
	// Release resources
	delete screen;
	if (mirrorTexture)
		ovr_DestroyMirrorTexture(session, mirrorTexture);
	for (int eye = 0; eye < 2; ++eye)
	{
		delete pEyeRenderTexture[eye];
		delete pEyeDepthBuffer[eye];
	}
	DIRECTX.ReleaseDevice();
	ovr_Destroy(session);
}

void UpdateTexture(cv::Mat input)
{
	screen->Fill->Tex->SetTextureMat(input);
}

bool Initalize_VR(HINSTANCE hinst, unsigned int output_width, unsigned int output_height)
{
	texture_width = output_width;
	texture_height = output_height;

	// Initializes LibOVR, the Rift, and the main rendering loop
	ovrResult result = ovr_Initialize(nullptr);
	if (!OVR_SUCCESS(result)){
		MessageBoxA(NULL, "Failed to initialize libOVR.", "VR Display", MB_ICONERROR | MB_OK);
		return false;
	}

	if (!DIRECTX.InitWindow(hinst, L"VR Display")){
		MessageBoxA(NULL, "Failed to open window.", "VR Display", MB_ICONERROR | MB_OK);
		return false;
	}

	return SetupMainLoop();
}