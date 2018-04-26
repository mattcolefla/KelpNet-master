Real ForwardActivate(Real gpuY)
{
	return 1.0 / (1.0 + exp(-gpuY));
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpuY <= 0.0 ? 0.0 : gpugX;
}
