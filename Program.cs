// Ported from Austin Eng's WebGPU Boids sample (https://austin-eng.com/webgpu-samples/samples/computeBoids)

using MoonWorks;
using MoonWorks.Graphics;
using MoonWorks.Math;
using System.IO;
using System.Runtime.InteropServices;

class Program : Game
{
	public static void Main(string[] args)
	{
		Program p = new Program(
			new WindowCreateInfo("MoonWorks Boids", 640, 480, ScreenMode.Windowed),
			PresentMode.FIFO,
			60,
			true
		);
		p.Run();
	}

	const int NUM_PARTICLES = 1500;

	GraphicsPipeline renderPipeline;
	Buffer spriteVertexBuffer;
	Buffer spriteIndexBuffer;

	ComputePipeline computePipeline;
	Buffer[] particleBuffers;

	[StructLayout(LayoutKind.Sequential)]
	struct SimParams
	{
		public float DeltaT;
		public float Rule1Distance;
		public float Rule2Distance;
		public float Rule3Distance;
		public float Rule1Scale;
		public float Rule2Scale;
		public float Rule3Scale;
		public int ParticleCount;
	}

	[StructLayout(LayoutKind.Sequential)]
	struct Particle
	{
		public Vector2 Pos;
		public Vector2 Vel;
	}

	[StructLayout(LayoutKind.Sequential)]
	struct PositionVertex
	{
		public Vector2 Position;

		public PositionVertex(float x, float y)
		{
			Position = new Vector2(x, y);
		}
	}

	public Program(WindowCreateInfo windowCreateInfo, PresentMode presentMode, int fps, bool debugMode)
		: base(windowCreateInfo, presentMode, fps, debugMode)
	{
		string baseContentPath = Path.Combine(
			System.AppDomain.CurrentDomain.BaseDirectory,
			"Content"
		);

		CommandBuffer cmdbuf = GraphicsDevice.AcquireCommandBuffer();

		// Render Pipeline
		ShaderModule spriteVertShaderModule = new ShaderModule(
			GraphicsDevice,
			Path.Combine(baseContentPath, "sprite_vert.spv")
		);

		ShaderModule spriteFragShaderModule = new ShaderModule(
			GraphicsDevice,
			Path.Combine(baseContentPath, "sprite_frag.spv")
		);

		renderPipeline = new GraphicsPipeline(
			GraphicsDevice,
			new GraphicsPipelineCreateInfo
			{
				AttachmentInfo = new GraphicsPipelineAttachmentInfo(
					new ColorAttachmentDescription(
						GraphicsDevice.GetSwapchainFormat(Window),
						ColorAttachmentBlendState.None
					)
				),
				DepthStencilState = DepthStencilState.Disable,
				MultisampleState = MultisampleState.None,
				PrimitiveType = PrimitiveType.TriangleList,
				ViewportState = new ViewportState((int) Window.Width, (int) Window.Height),
				RasterizerState = RasterizerState.CW_CullNone,
				VertexInputState = new VertexInputState
				{
					VertexBindings = new VertexBinding[]
					{
						new VertexBinding
						{
							Binding = 0,
							InputRate = VertexInputRate.Instance,
							Stride = (uint) Marshal.SizeOf<Particle>()
						},
						new VertexBinding
						{
							Binding = 1,
							InputRate = VertexInputRate.Vertex,
							Stride = (uint) Marshal.SizeOf<PositionVertex>()
						}
					},
					VertexAttributes = new VertexAttribute[]
					{
						VertexAttribute.Create<Particle>("Pos", 0),
						VertexAttribute.Create<Particle>("Vel", 1),
						VertexAttribute.Create<PositionVertex>("Position", 2, 1)
					}
				},
				VertexShaderInfo = GraphicsShaderInfo.Create(spriteVertShaderModule, "main", 0),
				FragmentShaderInfo = GraphicsShaderInfo.Create(spriteFragShaderModule, "main", 0)
			}
		);

		// Vertex Buffer
		spriteVertexBuffer = Buffer.Create<PositionVertex>(
			GraphicsDevice,
			BufferUsageFlags.Vertex,
			3
		);
		cmdbuf.SetBufferData<PositionVertex>(
			spriteVertexBuffer,
			new PositionVertex[]
			{
				new PositionVertex(-0.01f, -0.02f),
				new PositionVertex(0.01f, -0.02f),
				new PositionVertex(0.0f, 0.02f)
			}
		);

		// Index Buffer
		spriteIndexBuffer = new Buffer(
			GraphicsDevice,
			BufferUsageFlags.Index,
			sizeof(ushort) * 3
		);
		cmdbuf.SetBufferData<ushort>(
			spriteIndexBuffer,
			new ushort[] { 0, 1, 2 }
		);

		// Compute Pipeline
		ShaderModule computeShaderModule = new ShaderModule(
			GraphicsDevice,
			Path.Combine(baseContentPath, "updateSprites.spv")
		);

		computePipeline = new ComputePipeline(
			GraphicsDevice,
			ComputeShaderInfo.Create<SimParams>(computeShaderModule, "main", 2, 0)
		);

		// Randomly generate initial particle data
		System.Random random = new System.Random();
		Particle[] initialParticleData = new Particle[NUM_PARTICLES];
		for (int i = 0; i < initialParticleData.Length; i++)
		{
			initialParticleData[i].Pos = new Vector2(
				2 * (random.NextSingle() - 0.5f),
				2 * (random.NextSingle() - 0.5f)
			);
			initialParticleData[i].Vel = new Vector2(
				2 * (random.NextSingle() - 0.5f) * 0.1f,
				2 * (random.NextSingle() - 0.5f) * 0.1f
			);
		}

		// Create and populate the compute buffers
		particleBuffers = new Buffer[2];
		for (int i = 0; i < particleBuffers.Length; i++)
		{
			particleBuffers[i] = Buffer.Create<Particle>(
				GraphicsDevice,
				BufferUsageFlags.Compute | BufferUsageFlags.Vertex,
				NUM_PARTICLES
			);
			cmdbuf.SetBufferData<Particle>(
				particleBuffers[i],
				initialParticleData
			);
		}

		GraphicsDevice.Submit(cmdbuf);
	}

	protected override void Update(System.TimeSpan dt) { }

	int t = 0; // Used for alternating between particle buffers
	protected override void Draw(System.TimeSpan dt, double alpha)
	{
		CommandBuffer cmdbuf = GraphicsDevice.AcquireCommandBuffer();
		Texture? swapchainTex = cmdbuf.AcquireSwapchainTexture(Window);

		if (swapchainTex != null)
		{
			// Compute
			cmdbuf.BindComputePipeline(computePipeline);
			uint computeUniformOffset = cmdbuf.PushComputeShaderUniforms<SimParams>(
				new SimParams
				{
					DeltaT = 0.04f,
					Rule1Distance = 0.1f,
					Rule2Distance = 0.025f,
					Rule3Distance = 0.025f,
					Rule1Scale = 0.02f,
					Rule2Scale = 0.05f,
					Rule3Scale = 0.005f,
					ParticleCount = NUM_PARTICLES
				}
			);
			cmdbuf.BindComputeBuffers(
				particleBuffers[t % 2],
				particleBuffers[(t + 1) % 2]
			);
			cmdbuf.DispatchCompute(
				(uint) System.Math.Ceiling(NUM_PARTICLES / 64.0),
				1,
				1,
				computeUniformOffset
			);

			// Render
			cmdbuf.BeginRenderPass(
				new ColorAttachmentInfo(
					swapchainTex,
					Color.CornflowerBlue
				)
			);
			cmdbuf.BindGraphicsPipeline(renderPipeline);
			cmdbuf.BindVertexBuffers(
				0,
				new BufferBinding(particleBuffers[(t + 1) % 2], 0),
				new BufferBinding(spriteVertexBuffer, 0)
			);
			cmdbuf.BindIndexBuffer(
				spriteIndexBuffer,
				IndexElementSize.Sixteen
			);
			cmdbuf.DrawInstancedPrimitives(0, 0, 1, NUM_PARTICLES, 0, 0);
			cmdbuf.EndRenderPass();
		}

		GraphicsDevice.Submit(cmdbuf);
		t++;
	}

	protected override void OnDestroy() { }
}
