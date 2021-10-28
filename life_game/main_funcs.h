#pragma once

void mouseMotionEvent(float& offset_x, float& offset_y, SDL_MouseMotionEvent& mouse)
{
	if (mouse.state & SDL_BUTTON_MIDDLE)
	{
		offset_x += mouse.xrel / static_cast<float>(SCALE.x);
		offset_y += mouse.yrel / static_cast<float>(SCALE.y);
	}
}

void windowResized(int& w, int& h, SDL_WindowEvent& event_)
{
	w = event_.data1;
	h = event_.data2;
}

void scaleWithWheel(pnd::Point2& origin, pnd::Point2& scale, SDL_MouseWheelEvent& event_)
{
	float scale_ = (event_.y > 0 ? 2.0f : 0.5f);
	if (scale_ == 2.0f && scale.w < 100 || scale_ == 0.5f)
	{
		pnd::Point2 tmp_scale = scale;

		scale = pnd::scale2(&origin, &scale, scale_);

		int current_x;
		int current_y;
		SDL_GetMouseState(&current_x, &current_y);


		if (tmp_scale.w != 1 || scale.w != 1)
		{
			if (event_.y > 0) tmp_scale = scale;

			float one_y = -event_.y / abs(event_.y);
			OFFSET_X += one_y * current_x / static_cast<float>(tmp_scale.x);
			OFFSET_Y += one_y * current_y / static_cast<float>(tmp_scale.y);
		}
	}
}

void killCell(Game& gm, SDL_MouseButtonEvent& event_)
{
	if (event_.button & SDL_BUTTON_LEFT)
	{
		float current_x = (event_.x / SCALE.x - OFFSET_X) / SHEAR_SIZE;
		float current_y = (event_.y / SCALE.y - OFFSET_Y) / SHEAR_SIZE;
		if (current_x < static_cast<int>(current_x) + RECT_SIZE / SHEAR_SIZE && current_y < static_cast<int>(current_y) + RECT_SIZE / SHEAR_SIZE && current_x >= 0 && current_y >= 0)
			gm.setCell({ (unsigned int)current_x, (unsigned int)current_y });
	}
}

void keyDown(bool& play, bool& one_beat, bool& show_field, int& speed, SDL_KeyboardEvent& event_)
{
	switch (event_.keysym.sym)
	{
	case SDLK_r:
	{
		play = !play;
	}
	break;
	case SDLK_d:
	{
		one_beat = true;
	}
	break;
	case SDLK_MINUS:
	{
		if (speed > 1)
			speed -= 2;
	}
	break;
	case SDLK_EQUALS:
	{
		if (speed <= 33)
			speed += 2;
	}
	break;
	case SDLK_q:
	{
		show_field = !show_field;
	}
	break;
	}
}

void oneBeat(Game& gm, bool& one_beat, bool play)
{
	if (play)
	{
		gm.tick();
	}
	else if (one_beat)
	{
		gm.tick();
		one_beat = false;
	}
}

void setFirstGeneration(Game& gm)
{
	gm.setCell({ 500, 500 });
	gm.setCell({ 500, 501 });
	gm.setCell({ 500, 502 });
	gm.setCell({ 500, 503 });
	gm.setCell({ 500, 504 });
	gm.setCell({ 500, 505 });
	gm.setCell({ 500, 506 });
	gm.setCell({ 500, 507 });
	gm.setCell({ 500, 509 });
	gm.setCell({ 500, 510 });
	gm.setCell({ 500, 511 });
	gm.setCell({ 500, 512 });
	gm.setCell({ 500, 513 });
	gm.setCell({ 500, 517 });
	gm.setCell({ 500, 518 });
	gm.setCell({ 500, 519 });
	gm.setCell({ 500, 526 });
	gm.setCell({ 500, 527 });
	gm.setCell({ 500, 528 });
	gm.setCell({ 500, 529 });
	gm.setCell({ 500, 530 });
	gm.setCell({ 500, 531 });
	gm.setCell({ 500, 532 });
	gm.setCell({ 500, 534 });
	gm.setCell({ 500, 535 });
	gm.setCell({ 500, 536 });
	gm.setCell({ 500, 537 });
	gm.setCell({ 500, 538 });

}