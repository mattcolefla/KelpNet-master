Real ForwardActivate(Real gpuY)
{
	return (Real)(1.2567348023993685 * ((log(gpuY + sqrt((gpuY * gpuY) + 1)) + 1.0) * 0.5));
            
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpugX * (1 - gpuY * gpuY);
}
