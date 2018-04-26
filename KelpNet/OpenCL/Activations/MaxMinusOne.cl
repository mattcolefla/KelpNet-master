Real ForwardActivate(Real gpuY)
{
	Real y;
    if (gpuY > -1)
    {
        y = gpuY;
    }
    else
    {
        y = -1;
    }
    return y;
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpugX * (1 - gpuY * gpuY);
}
