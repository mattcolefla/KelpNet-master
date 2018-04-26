Real ForwardActivate(Real gpuY)
{
	Real alpha = 1.6732632423543772848170429916717;
    Real scale = 1.0507009873554804934193349852946;

    Real y;
    if (gpuY >= 0)
    {
        y = scale * gpuY;
    }
    else
    {
        y = scale * (alpha * exp(gpuY) - alpha);
    }

    return y;
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpugX * (1 - gpuY * gpuY);
}
