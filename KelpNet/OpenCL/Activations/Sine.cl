Real ForwardActivate(Real gpuY)
{
	return sin(2.0 * gpuY);
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpugX * (1 - gpuY * gpuY);
}
